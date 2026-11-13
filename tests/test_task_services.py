"""Sidecars (compose services): declaration checks, fingerprint, and the runtime's
network + sidecar lifecycle against a recording Docker client."""

from types import SimpleNamespace

import pytest
from episode_fixtures import IMAGE
from fake_episode_env import extract, grade, prepare

from reliquary_sandbox.episode_task import SandboxTask, TaskLimits, TaskService
from reliquary_sandbox_service.episodes.registry import ResolvedTask, task_fingerprint
from reliquary_sandbox_service.task_runtime import BoxSpec, InfraFault, TaskBoxRuntime

GIB = 1024**3
REDIS = "redis@sha256:" + "1" * 64


def task(services=(), memory=2 * GIB):
    return SandboxTask(image=IMAGE, workdir="/w", tools=("bash",), prepare=prepare, extract=extract,
                       grade=grade, services=services, limits=TaskLimits(memory_bytes=memory))


def test_declaration_checks():
    with pytest.raises(ValueError):
        TaskService("Bad", REDIS)
    with pytest.raises(ValueError):
        TaskService("redis", "redis:7")                      # not pinned
    with pytest.raises(ValueError):
        task((TaskService("a", REDIS), TaskService("a", REDIS)))
    with pytest.raises(ValueError):
        task((TaskService("redis", REDIS, memory_bytes=2 * GIB),))   # leaves < 256 MiB for the box
    assert task((TaskService("redis", REDIS),)).services[0].name == "redis"


def test_fingerprint_changes_only_with_services():
    def fp(t):
        return task_fingerprint(ResolvedTask(task=t, env_package="p==1"))
    assert fp(task()) == fp(task())
    assert fp(task()) != fp(task((TaskService("redis", REDIS),)))


class _Obj(SimpleNamespace):
    def start(self):
        self.calls.append(("start", self.name))

    def remove(self, force=False):
        self.calls.append(("remove", self.name))

    def reload(self):
        pass

    def connect(self, container):
        self.calls.append(("connect", self.name))


class _Client:
    def __init__(self, fail_service=False):
        self.calls, self.fail_service = [], fail_service
        self.containers = SimpleNamespace(create=self._create, list=self._list)
        self.networks = SimpleNamespace(create=self._net, get=self._get_net)
        self.images = SimpleNamespace(get=lambda ref: SimpleNamespace(id="img-" + ref[:5]))
        self.api = SimpleNamespace(create_endpoint_config=lambda aliases: {"Aliases": aliases})
        self.created = []

    def _create(self, **kwargs):
        if self.fail_service and kwargs["name"].endswith("-svc-redis"):
            raise RuntimeError("boom")
        self.created.append(kwargs)
        net = kwargs.get("network")
        attrs = {"NetworkSettings": {"Networks": {net: {"IPAddress": "172.30.0.2"}}}} if net else {}
        status = "running" if kwargs["name"].endswith("-svc-redis") else "exited"
        return _Obj(name=kwargs["name"], calls=self.calls, status=status, attrs=attrs)

    def _list(self, all=True, filters=None):
        return [_Obj(name=c["name"], calls=self.calls) for c in self.created
                if c["labels"].get("reliquary.episode.managed") == "sidecar"]

    def _net(self, name, **kwargs):
        self.calls.append(("network", name, kwargs["internal"]))

    def _get_net(self, name):
        return _Obj(name=name, calls=self.calls)


def runtime(client):
    rt = TaskBoxRuntime.__new__(TaskBoxRuntime)
    rt.settings = SimpleNamespace(episode_images=[IMAGE, REDIS], docker_runtime="runsc",
                                  episode_allow_unquoted_disk=False, episode_egress_network="rq-egress",
                                  state_dir="/tmp")
    rt.client, rt.namespace = client, "t"
    rt._check_helper = lambda: None
    rt._etc_mounts = lambda box_id, hosts=None: ["etc", hosts]
    rt._cpu_last, rt._box_users, rt._helper_retries = {}, {}, {}
    return rt


def spec(**kw):
    values = dict(image=IMAGE, workdir="/w", env={}, user="0", memory_bytes=2 * GIB,
                  nano_cpus=2_000_000_000, pids=512, disk_bytes=GIB, lifetime_s=60,
                  services=(TaskService("redis", REDIS, memory_bytes=512 * 1024**2, cpus=4),))
    values.update(kw)
    return BoxSpec(**values)


def test_sidecars_on_a_private_network_then_the_box():
    client = _Client()
    rt = runtime(client)
    with pytest.raises(InfraFault):      # the fake box never runs: torn down with its services
        rt.create("b1", "e1", spec(network="rq-egress"))
    net = rt._service_network("b1")
    assert len(net) <= 15 and client.calls[0] == ("network", net, True)
    svc, box = client.created
    assert svc["networking_config"] == {net: {"Aliases": ["redis"]}} and svc["network"] == net
    assert svc["runtime"] == "runsc" and svc["cap_drop"] == ["ALL"]
    assert svc["nano_cpus"] == 2_000_000_000                 # capped at the box's CPUs
    assert svc["storage_opt"] == {"size": str(2 * GIB)} and svc["labels"]["reliquary.episode.box"] == "b1"
    assert box["network_mode"] == net and box["mounts"] == ["etc", {"redis": "172.30.0.2"}]
    assert box["mem_limit"] == 2 * GIB - 512 * 1024**2
    assert ("connect", "rq-egress") in client.calls          # online box: egress joined too
    assert ("remove", svc["name"]) in client.calls and ("remove", net) in client.calls


def test_a_failed_sidecar_is_ours_and_cleaned():
    client = _Client(fail_service=True)
    with pytest.raises(InfraFault):
        runtime(client).create("b2", "e2", spec())
    assert client.created == [] and ("remove", runtime(client)._service_network("b2")) in client.calls


def test_unapproved_service_image_refused():
    with pytest.raises(ValueError):
        runtime(_Client()).create("b3", "e3", spec(services=(
            TaskService("redis", "other@sha256:" + "2" * 64),)))
