# Hybrid-MovieLens-Recommender

This project explores a hybrid recommender systems by combining collaborative filtering with demographic and genre-based features using the MovieLens 100k dataset. 
This was done as an academic exploration to learn, rather than build a production-ready system. 

The model predicted ratings with reasonable accuracy (MAE ≈ 0.90, RMSE ≈ 1.12), but recommendation quality was poor (Precision ≈ 0.12, Recall ≈ 0.01, NDCG ≈ 0.14). I suspect conceptual improvements could have been made through better ranking methods, integrating demographic and collaborative signals into explanations, and diversifying features beyond genres. Furthermore, I believe improvement to the model would require longer training with a validation set, lower learning rate, smaller batch size, and ranking-oriented loss functions such as BPR.


[A Hybrid Recommendation System Leveraging Collaborative Filtering and Demographic Features Paper](https://github.com/user-attachments/files/22733418/A_Hybrid_Recommendation_System_Leveraging_Collaborative_Filtering_and_Demographic_Features.pdf)

[Hybrid Recommendation Presentation Slides](https://github.com/user-attachments/files/22733419/Hybrid_Recommendation_Presentation.pdf)
