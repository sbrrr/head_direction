HDcoactivity contains the code for identifying the preferred head direction of each head direction cell, computing which cells are significantly coactive (note: each head direction cell is expected to be coactive with it self and HD cells with neighbouring preferred angles), sorts the cells by angle in a matrix that shows which pairs of coactive cells are active at each time point (t measured in milliseconds) and prints a plot of the coactivity matrix, actual head direction and the % of coactive cells active for each time point. 

The sequence of images can be compiled into a video using ffmpeg. "img sequence to video ffmpeg.txt" contains the code to do so and some relevant information on ffmpeg. The framerate of the video was quadrupled before publishing the video using the code in "ffmpeg change playback speed.txt". 
Link to the video:
https://youtu.be/JkUw0rLAWc4
