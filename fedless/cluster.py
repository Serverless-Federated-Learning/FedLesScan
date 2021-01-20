from dataclasses import dataclass


@dataclass
class Node:
    name: str
    domain: str
    port: int = 80

    @property
    def address(self) -> str:
        return f"{self.domain}:{self.port}"
