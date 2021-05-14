FROM fedless-flower

COPY ./main.py main.py
CMD ["python", "main.py"]
#python server.py --server_address <YOUR_SERVER_IP:PORT> --rounds 3 --min_num_clients 1 --min_sample_size 1 --model ResNet18
