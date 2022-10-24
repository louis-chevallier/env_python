
# bash -ic 'source buildenv.sh && build_python_env_docker install'; rm -fr conda116_1.11.0_0.7.0;

start :
	python server.py


docker_build :
	sudo docker build --no-cache --progress=auto -t jitdeep/cara .

