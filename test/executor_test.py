from time import sleep
from multi_tenant.DockerExecutor import DockerExecutor

DockerExecutor.setup(force_rebuild=False)

executor = DockerExecutor()
a, b = executor._execute_code("print('Hello, world!')")

print("DockerExecutor setup complete")