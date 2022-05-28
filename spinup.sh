export FIXUID=$(id -u) 
export FIXGID=$(id -g) 
docker-compose build 
docker-compose run -u root mhsecond