# IDW
The **Inverse Distance Weighted (IDW) Method** in Python for Land Fog Detection based on visibility data from Observation Station, that categorized land fog into six leveles (0-5). Two types of visualization for the results are provided: a scatter plot with levels and an interpolated image.

## Requirements
```
Python  >= 3.9.0
cnmaps  >= 1.1.7
cartopy >= 0.22.0
geojson >= 3.0.1
```

## IDW Method
For a certain point that requires interpolation:
- Identify the top-k nearesr neighboring points, calculate the distances and record the indexes.
- Determine whether these k points are all within the maximum distance set. Only keep the valid points.
- If there are no points that meet the distance requirement, skip the interpolation.
- If two points are very close to each other, use the strategy of duplicating the point.

## Visility Data
The Observation station files are in `.000 format`, but since this is unpublished data for the ongoing project, we cannot provide the corresponding data file in our repository. However, other publicly available data can be obtained and used for IDW code as long as it is read into a `Pandas array`.

## Visualization
The visualization results (2022-12-10 12:00 UTC, Scatter-Left, IDW-Right) are shown in the following figure:
<img width="907" alt="image" src="https://github.com/kaka0910/IDW/assets/23305257/90d5c94d-fac8-40ac-acd3-f9791dc3bdec">

