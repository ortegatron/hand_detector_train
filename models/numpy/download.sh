#!/bin/bash

echo "download videos"

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget --continue $( extract_download_url http://www.mediafire.com/file/ropayv77vklvf56/openpose_coco.npy ) -O openpose_coco.npy
