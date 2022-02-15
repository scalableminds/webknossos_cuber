import cluster_tools


def square(n):
    return n * n


def test_kubernetes():
    with cluster_tools.get_executor(
        "kubernetes",
        job_resources={
            "memory": "100M",
            "python_executable": "python",
            "image": "scalableminds/cluster_tools:latest",
            "node_selector": {},
            "namespace": "cluster-tools",
        },
        debug=True,
    ) as exec:
        assert list(exec.map(square, [n + 2 for n in range(2)])) == [4, 9]


if __name__ == "__main__":
    test_kubernetes()
