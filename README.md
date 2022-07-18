# GridSmart Real-time Detection

This GUI application is developed for roadside fisheye camera perception.

## Prerequisite

* [OpenCV](https://pypi.org/project/opencv-python/)
* [BGSLibrary](https://github.com/andrewssobral/bgslibrary)
* [TkInter](https://docs.python.org/3/library/tkinter.html)

## Streaming

To start, input the RTSP address in the text box and click **start streaming** button.

## Background Initialization 

Then run the **background initialization function**. There are three options:

1. Auto Init: Input duration of the initialization process (10s by default, usually need 300s for good performance), and then click **Auto Init** button.
2. Zone BG Update: If you want to only update certain zones, you may use this function. Input duration of the initialization process (10s by default), select zone in the text box (can be multi-selected), and click **Zone BG Update** button.
3. Load background in the database: Click **BG Load** button.

## Background Maintaining

After initialization, click **BG Maintaining** button to enable **dynamic background maintaining function**. You can save maintained background by clicking the **Save BG** button.

## Detection

You can start the detection as long as you get a good background. Clicking the **Start Detection** button to enable detection. You will see detection result visualized on the CMM canvas. You can save detected images and/or CSV data by clicking the corresponding button.
