# PYNQ board details to run server code

ssh into the board: "ssh -X -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.1.224" password: xilinx

code sits in /board_ui/server.py

To run the code: "python3 server.py"

Any changes in the board server code will require a restart of the board server.

# For local machine to run the UI

npm.cmd run dev

First ssh into the board and run the board server code

In new terminal, run "ssh -L 8000:127.0.0.1:8000 -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.1.224" to tunnel the board to your local machine, password: xilinx

# To run the UI on internet

First ssh into the board and run the board server code

In new terminal, run "ssh -L 8000:127.0.0.1:8000 -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.1.224" to tunnel the board to your local machine, password: xilinx

In another new terminal, run "ngrok http 8000" to get the https:// link from ngrok and paste it in the UI board URL field.