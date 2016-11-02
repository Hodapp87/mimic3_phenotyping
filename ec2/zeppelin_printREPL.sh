#!/bin/bash

cd /

cat <<EOF | patch -p0 -b -t
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

EOF
