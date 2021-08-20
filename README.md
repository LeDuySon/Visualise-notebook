# Visualise-notebook

## Info
- task-count-people.ipynb: Show histogram about number of people in multiple videos (TrackHuman UET lab)


## Postprocess 
- Folder structure:

```
.
├── get_invalid_area.py
├── __init__.py
├── output_post_processing.py
├── test.py

```

- Explain folder :
    - get_invalid_area.py: Draw rectangles on region you want to ignore -> save to file in static/ignore_region
    - output_post_processing.py: Filter boxes that in ignore_region
    - test.py: Show video and boxes, just for testing our filter scripts 

- Notes:
    - Only work with yolo format 
    - If u want to use other formats, comment out scaled_yolo_coord() func
