#!/usr/bin/env bash

set -e

# Build and push server image
docker build -f server.Dockerfile -t "flower-server" .
docker tag "flower-server" andreasgrafberger/flower:server
docker push andreasgrafberger/flower:server

# Build and push client image
docker build -f client.Dockerfile -t "flower-client" .
docker tag "flower-client" andreasgrafberger/flower:client
docker push andreasgrafberger/flower:client

#ssh_host="invasic"
#server_ip="172.24.65.11"
ssh_host="lrz-master"
server_ip="138.246.233.67"
port="31532"
server_address="$server_ip:$port"
server_cpus="4"     #"16"
server_memory="16g" # "32g"
rounds=2000
min_num_clients=25
num_clients_total=70

client_cpus=2.0
client_memory="8g"
dataset="shakespeare"
batch_size=10
epochs=1
optimizer="SGD"
lr=0.8

declare -a ssh_hosts=("invasic" "lrz-4-master" "lrz-4-worker-1" "lrz-4-worker-2" "lrz-4-worker-3" "lrz-4-worker-4" "lrz-4-worker-5") # "lrz-master" )

echo "Removing running container if it exists..."
ssh "$ssh_host" 'docker stop fl-server' ||
  true

ssh "$ssh_host" "docker pull andreasgrafberger/flower:server"
session_id="$RANDOM"
run_cmd="docker run --rm -p $port:$port --name fl-server \
-e https_proxy=\$http_proxy \
--cpus $server_cpus --memory $server_memory --memory-swap $server_memory \
andreasgrafberger/flower:server --rounds $rounds --min-num-clients $min_num_clients --dataset=$dataset"
echo "Starting server, results are stored in fedless_$session_id.out and fedless_$session_id.err"
# shellcheck disable=SC2029
ssh "$ssh_host" "nohup $run_cmd > fedless_$session_id.out 2> fedless_$session_id.err < /dev/null &"

echo "Deploying and starting clients..."

current_partition=0

echo "Making sure all client machines have the latest docker images and no running clients"
for ssh_host in "${ssh_hosts[@]}"; do
  ssh "$ssh_host" "docker pull andreasgrafberger/flower:client"
  ssh "$ssh_host" "docker ps | grep andreasgrafberger/flower:client | cut -d ' ' -f 1 | xargs -r docker stop"
  ssh "$ssh_host" "sudo usermod -aG docker \$USER " || true
done

echo "Starting clients..."
for ssh_host in "${ssh_hosts[@]}"; do
  if [[ $current_partition -ge $num_clients_total ]]; then
    break
  fi
  echo "Starting clients on $ssh_host"
  num_cores_available=$(ssh "$ssh_host" cat /proc/cpuinfo | grep -c processor)
  # shellcheck disable=SC2004
  num_potential_clients=$(($num_cores_available / $client_cpus))
  num_potential_clients=${num_potential_clients%.*}
  echo "Host $ssh_host has $num_cores_available cores available. Assigning $num_potential_clients client functions \
with $client_cpus cores each to it..."

  for ((i = 1; i <= num_potential_clients; i++)); do
    run_cmd="docker run --rm \
--network host \
--cpus $client_cpus \
--memory $client_memory \
--memory-swap $client_memory \
-e https_proxy=\$http_proxy \
-e no_proxy=$server_ip \
andreasgrafberger/flower:client \
--server $server_address \
--dataset $dataset \
--partition $current_partition \
--batch-size $batch_size \
--epochs $epochs \
--optimizer $optimizer \
--lr $lr \
--clients-total $num_clients_total"
    echo "($ssh_host) $run_cmd"
    # shellcheck disable=SC2029
    ssh "$ssh_host" "nohup $run_cmd > fedless_${session_id}_$current_partition.out 2> fedless_${session_id}_$current_partition.err < /dev/null &"
    current_partition=$((current_partition + 1))
  done
done

if ((current_partition >= num_clients_total)); then
  echo "Successfully deployed all clients"
else
  echo "WARNING: Tried to deploy client partition ($current_partition / $num_clients_total) but no compute left..."
fi
