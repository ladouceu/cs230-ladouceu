# cs230-ladouceu
Repo for final project of Stanford course in Deep Learning: CS230

## to do
priorities
- make a siamese network that works just with th images [done]
- make my siamese network train in parallel and try it out in colab with GPU (*easy*, takes about 45 min per epoch)
- then test out my model by submitting my results to MOTchallenge (requires that I build a whole new pipeline), then I will have some initial results 
- make a new train dataset using generator that would be more accurate
- add current position to my data in siamese model (this requires that I use the next step as my positive and negative but different ids and same video, maybe use the closest images as examples for negative), this is my first improvement over the basic model

less critical
- improve the pipeline as suggested in the todo (using the experimental)
- try to add the 4 last location as features
- try to add the lstm for location



maybe the next pipeline that I build I could simply make a dataset from generator and
have my generator iterate over the video, frames, ids, it would be simpler.