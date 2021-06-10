filename=bear
# ffmpeg -i "$filename"15fps.mp4 -i "$filename"30fps_gt.mp4 -i "$filename"30fps_interpolate.mp4 -filter_complex "nullsrc=size=1708x960 [base]; [0:v] setpts=PTS-STARTPTS, scale=854x480 [upperleft]; [1:v] setpts=PTS-STARTPTS, scale=854x480 [upperright]; [2:v] setpts=PTS-STARTPTS, scale=854x480 [lowerright]; [base][upperleft] overlay=shortest=1 [tmp1]; [tmp1][upperright] overlay=shortest=1:x=854 [tmp2]; [tmp2][lowerright] overlay=shortest=1:x=854:y=480" -c:v libx264 output"$filename".mp4

# ffmpeg -f concat -safe 0 -i videolist.txt -c copy upload.mp4
gtfilename="$filename"30fps_gt.mp4
interfilename="$filename"30fps_interpolate.mp4
mpv --lavfi-complex="[vid1][vid2][vid3]hstack=inputs=3[vo];[aid1][aid2][aid3]amix=inputs=3[ao]" "$filename"15fps.mp4 --external-files="./$gtfilename:./$interfilename" --keep-open --title="Left:15fps;              Middle:30fps Ground Truth;                      Right:30fps Interpolated"
