#!/bin/bash

cd /

cat <<EOF | sudo /usr/bin/patch -p0 -b -t
*** /etc/zeppelin/conf/interpreter.json_bak	2016-11-02 14:44:45.396942971 +0000
--- /etc/zeppelin/conf/interpreter.json	2016-11-02 14:49:17.997687967 +0000
***************
*** 5,10 ****
--- 5,11 ----
        "name": "spark",
        "group": "spark",
        "properties": {
+         "zeppelin.spark.printREPLOutput": "true",
          "spark.yarn.jar": "",
          "master": "yarn-client",
          "zeppelin.spark.maxResult": "1000",
***************
*** 30,36 ****
            "name": "sql"
          }
        ],
!       "dependencies": [],
        "option": {
          "remote": true,
          "perNoteSession": false,
--- 31,42 ----
            "name": "sql"
          }
        ],
!       "dependencies": [
!         {
!           "groupArtifactVersion": "com.cloudera.sparkts:sparkts:0.4.0",
!           "local": false
!         }
!       ],
        "option": {
          "remote": true,
          "perNoteSession": false,
EOF

sudo /usr/lib/zeppelin/bin/zeppelin-daemon.sh restart
