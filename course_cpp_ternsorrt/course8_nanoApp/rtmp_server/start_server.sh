#ffserver -f server1.conf &
#ffserver -f server2.conf &
#ffserver -f server3.conf &
#ffserver -f server4.conf &
#ffmpeg -i 1.mp4 -vcodec libx264 -tune zerolatency -crf 18 http://localhost:1234/feed1.ffm &
#ffmpeg -i 2.mp4 -vcodec libx264 -tune zerolatency -crf 18 http://localhost:1235/feed2.ffm &
#ffmpeg -i 3.mp4 -vcodec libx264 -tune zerolatency -crf 18 http://localhost:1236/feed3.ffm &
#ffmpeg -i 4.mp4 -vcodec libx264 -tune zerolatency -crf 18 http://localhost:1237/feed4.ffm &
./rtsp-simple-server rtsp-simple-server-full.yml &
ffmpeg -re -stream_loop -1 -i test.mp4 -vcodec copy -acodec copy -b:v 5M -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live1.sdp &