#ffserver -f server1.conf &
#ffmpeg -i 1.mp4 -vcodec libx264 -tune zerolatency -crf 18 http://localhost:1234/feed1.ffm &

./rtsp-simple-server rtsp-simple-server-full.yml &
# ffmpeg -re -stream_loop -1 -i test.mp4 -vcodec copy -acodec copy -b:v 5M -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live1.sdp &

# camera -f v4l2 
ffmpeg -re -i /dev/video0 -r 25 \
    -c:v libx264 \
    -preset ultrafast -tune zerolatency \
    -b:v 5M -g 48 \
    -vf "scale=-1:360,format=yuv420p" \
    -c:a aac -b:a 128k \
    -ar 44100 -ac 2 \
    -f rtsp -rtsp_transport tcp rtsp://localhost:8554/camera

