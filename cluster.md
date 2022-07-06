## Connecting to the vpn

In order to connect to the CAMP cluster, you first need to connect the server via VPN. This can be done by using OpenVPN.

1- Download OpenVPN
2- Download attachment from CAMP cluster setup email (subject: CAMP Cluster Account Created)
3- When OpenVPN is installed you can right click to the OpenVPN logo on the task bar tray (more button alongside battery, wifi, sound icons) and select import. Then give downloaded attachment.
4- Username & password is the same as on the email sent.
5- OpenVPN can take some time to connect don't worry.

## Connection to the OpenPAI

This is probably easier than you imagined. To deploy some task on cluster or monitor existing tasks, it is not more then a webapp exeriments thanks to the OpenPAI.

1- Simply connect to the "master.garching.cluster.campar.in.tum.de" by any browser or 131.159.10.203
2- Since website does not have a valid SSL certificate (there is no need), some browsers can refuse to load the page. You can go advenced settings on those error pages and say let me in for this time. I wouldn't suggest disabling this feature alltogether.
3- You can login with the username & password sent to you via email
4- You can see overall loads on the servers on the Home tab. Sometimes it gets hard to schedule your tasks if the load is more than %70. I had to wait for one day for my task to be scheduled in one occasion.
5- You can submit a new job on submit job segment. Good part is when you select single job, command block is a linux command line. So you can write any command you wish. For our project though, you can find relevant scripts on scripts/kubernetes/ path. From this screen you can set up a request on GPU and CPU count (task will be scheduled according to this). There is no need to alter memory since it is already enough and it is not VRAM that you are requesting.
6- PLEASE DO NOT FORGET TO SELECT nfs-students ON THE DATA TAB ON CONFIG PAGE
7- PLEASE ADHERE TO THE COMMENT LINES ON THE KUBERNETES SCRIPTS
8- After you set up your task simply press on submit. You can also fine tune some settings by Edit YAML but this is very rarely necessary.

## Monitoring the tasks

You can monitor your deployed tasks over the jobs tab on OpenPAI. Simply select a task you have deployed and look at it's console outputs via stdout or error outputs via stderr. Keep in mind that those logs are not complete and only shows you recent ones instead of all. To see all of them please look at the weights & biases.

## Connecting to the NFS server

To upload run documents you need to connect to the NFS server. This is trivial for windows.

1- Open windows explorer (file manager)
2- Enter "\\10.8.0.1\workfiles" as path to the explorer when you are connected to the VPN as instructed above.
3- Navigate to the folder named your username
4- Enjoy

Please note that if username is asked you can try your username & password sent by email or login with TUM account. It didn't ask anything to me in that regard.