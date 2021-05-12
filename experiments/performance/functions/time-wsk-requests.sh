#!/usr/bin/env bash

curl -s -w "%{http_code} %{time_namelookup} %{time_connect} %{time_appconnect} %{time_pretransfer} %{time_starttransfer} %{time_total}\n" -o /dev/null "<URL>" >> write-out.log

curl -X POST -H "X-Require-Whisk-Auth: KM4qmVw/ADqA/EQ/dyBfipH4eTA1gEP/1dcEMpD3" \
  -w '\nEstablish Connection: %{time_connect}s\nTTFBReq: %{time_appconnect}s\nTTLBReq: %{time_pretransfer}s\nTTFBRes: %{time_starttransfer}s\nTotal: %{time_total}s\n' \
  -k https://138.246.233.67:31001/api/v1/web/guest/default/client.json \
  -d @request.json

#-w "%{http_code} %{time_namelookup} %{time_connect} %{time_appconnect} %{time_pretransfer} nTTFB %{time_starttransfer} %{time_total}\n" \
