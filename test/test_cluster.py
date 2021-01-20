from fedless.cluster import Node


def test_node_address_correct():
    node = Node("name", "192.160.178.1", port=80)
    assert node.address == "192.160.178.1:80"
