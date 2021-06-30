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

ssh_host="lrz-4xlarge"
server_ssh_host="lrz-4xlarge"
server_ip="138.246.233.207"
port="31532"
server_address="$server_ip:$port"
server_cpus="16"    #"4"
server_memory="32g" # "16g"
rounds=100
min_num_clients=100
num_clients_total=100

client_cpus=2.0
client_memory="8g"
dataset="mnist"
batch_size=10
epochs=5
optimizer="Adam"
lr=0.001

session_id="$RANDOM"

declare -a ssh_hosts=("invasic" "sk1" "sk2" "lrz-1" "lrz-2" "lrz-4xlarge" "lrz-3")
for session_id in "$RANDOM" "$RANDOM" "$RANDOM"; do
  for dataset in "mnist"; do
    for min_num_clients in 100; do # 75 50 25
      for epochs in 5 1; do # 1 10
        echo "Removing running container if it exists..."
        ssh "$server_ssh_host" 'docker stop fl-server' ||
          true

        ssh "$server_ssh_host" "docker pull andreasgrafberger/flower:server"
        run_cmd="docker run --rm -p $port:$port --name fl-server \
-e https_proxy=\$http_proxy \
--cpus $server_cpus --memory $server_memory --memory-swap $server_memory \
andreasgrafberger/flower:server --rounds $rounds --min-num-clients $min_num_clients --dataset=$dataset"
        ssh "$server_ssh_host" "mkdir -p flower-logs"
        exp_filename="flower-logs/fedless_${dataset}_${min_num_clients}_${num_clients_total}_${epochs}_${session_id}"
        echo "Starting server, results are stored in $exp_filename.out and $exp_filename.err"
        # shellcheck disable=SC2029
        ssh "$server_ssh_host" "nohup $run_cmd > $exp_filename.out 2> $exp_filename.err < /dev/null &"

        echo "Deploying and starting clients..."

        current_partition=0

        echo "Making sure all client machines have the latest docker images and no running clients"
        for ssh_host in "${ssh_hosts[@]}"; do
          ssh "$ssh_host" "docker pull andreasgrafberger/flower:client"
          ssh "$ssh_host" "docker ps | grep andreasgrafberger/flower:client | cut -d ' ' -f 1 | xargs -r docker stop"
          ssh "$ssh_host" "sudo usermod -aG docker \$USER " || true
          ssh "$ssh_host" "mkdir -p flower-logs"
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
            if [[ $current_partition -ge $num_clients_total ]]; then
              break
            fi
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
            ssh "$ssh_host" "nohup $run_cmd > ${exp_filename}_$current_partition.out 2> ${exp_filename}_$current_partition.err < /dev/null &"
            current_partition=$((current_partition + 1))
          done
        done

        if ((current_partition >= num_clients_total)); then
          echo "Successfully deployed all clients"
        else
          echo "WARNING: Tried to deploy client partition ($current_partition / $num_clients_total) but no compute left..."
        fi
        sleep 10
        while ssh "$server_ssh_host" "docker ps | grep andreasgrafberger/flower:server"; do
          echo "Not finished yet"
          sleep 10
        done
        #for ssh_host in "${ssh_hosts[@]}"; do
        #  if [ "$ssh_host" != "$server_ssh_host" ]; then
        #    scp "$ssh_host:*_*_*_*.out" "$server_ssh_host:flower-logs"
        #    scp "$ssh_host:*_*_*_*.err" "$server_ssh_host:flower-logs"
        #  fi
        #done
      done
    done
  done
done
