by: Minchan Kim, Chia Lee, Lily Weng

<p>
    <a href="https://github.com/m1nce/Describe-and-Refine" style="background-color:#1f63b9; color:white; padding:10px 15px; text-decoration:none; font-weight:bold; border-radius:5px;">
        Codebase ⚙️
    </a>
    <a href="https://drive.google.com/file/d/1hPwYXKKoe2vmegLnpAYZxFA5ZLDf6F1L/view?usp=drive_link" style="background-color:#e07b39; color:white; padding:10px 15px; text-decoration:none; font-weight:bold; border-radius:5px;">
        Report 📝
    </a>
    <a href="https://github.com/m1nce/Describe-and-Refine/blob/site/CLEAR%20Contextual%20Latent%20Embedding%20for%20Adaptive%20Representation.jpg" style="background-color:#8788d5; color:white; padding:10px 15px; text-decoration:none; font-weight:bold; border-radius:5px;">
        Poster 📜
    </a>
</p>

## Introduction

Understanding hidden neurons in vision models is key to improving interpretability. Describe-and-Dissect \textbf{(DnD)} is a training-free method that generates highly rated neuron explanations without labeled data or retraining. However, we believe that we could build on and even improve model performance by incorporating more machine learning techniques. We introduce Describe-and-Refine \textbf{(DnR)}, an effort to enhance \textbf{DnD} by adding several learning techniques to improve neuron interpretation. We introduce reinforcement learning by adding the option for users to input concepts, create custom scoring functions for measuring which concept best fits a neuron, and an iterative process for stable diffusion image generation and rating candidate concept accuracy to find the best concept fit for the highest activating generated images.

<center>
<table>
<tr>
    <td align="center">
        <img src="figure/DnD Pipeline.PNG" alt="DnD" width="350"/>
        <br>
        <b>(a) DnD</b>
    </td>
    <td align="center">
        <img src="figure/DnR Pipeline.PNG" alt="DnR" width="350"/>
        <br>
        <b>(b) DnR</b>
    </td>
</tr>
</table>
</center>

**Figure 1:** Describe-and-Dissect Versus Describe-and-Refine

## Objectives

\begin{itemize}
    \item Output more robust neuron concept explanations.
    \item Improve model performance and runtime efficiency.
    \item Reveal insight on black box neural network functionality in vision networks.
    \item Explore potential for reinforcement learning and user improvement in a label-free neuron descriptor.
\end{itemize}

## Methods

- **About Describe-and-Dissect (DnD) and Describe-and-Refine (DnR).**  
  - We implemented and tested **DnD** and **DnR** for neuron interpretability, focusing on **scoring methods, iterative feedback, and user-driven concept reinforcement**.  
  - Our experiments were conducted on the **ImageNet dataset** using **UCSD’s Data Science Machine Learning Platform (DSMLP)** on Datahub.  

- **DnD enables training-free neuron interpretation** by associating neuron activations with **textual concepts** and refining them through **iterative feedback using synthetic image generation and ranking**.  

- **DnR improves upon DnD by optimizing scoring mechanisms and iterative feedback.**  
  - Implemented **Top-K Log-Weighted Activation** and **Top-K Semantic Consistency** scoring to rank neurons effectively.  
  - Refined iterative scoring by comparing **concept scores** across multiple iterations, maintaining a master list for evaluation.  
  - Introduced **user-input reinforcement learning**, allowing for manual concept refinement before synthetic image generation.  

- **Testing setup and computational environment.**  
  - Conducted experiments on **UCSD’s DSMLP Kubernetes cluster**, leveraging **GPU acceleration** for efficient processing.  

- **Top-K Scoring Enhancements:**  
  - **Top-K Log-Weighted Activation:** Prioritizes strong activations while applying logarithmic scaling to balance extreme values.  
    - Formula:  
      \[
      S_{\text{log-k}} = \frac{1}{K} \sum_{i=1}^{K} \log(1 + a_{(i)})
      \]  
  - **Top-K Semantic Consistency:** Combines activation strength with **semantic alignment** using CLIP scores.  
    - Formula:  
      \[
      S_{\text{semantic-k}} = \alpha \cdot \left( \frac{1}{K} \sum_{i=1}^{K} a_{(i)} \right) + (1 - \alpha) \cdot \text{CLIP}(n)^2
      \]  

- **Iterative Feedback and Concept Validation.**  
  - **Synthetic images** were generated using **Stable Diffusion** based on extracted neuron concepts.  
  - Images were reintroduced to the network, validating their alignment with the original neuron activations.  

- **Final evaluation and impact.**  
  - Improved **interpretability** by refining neuron labels for **better human understanding**.  
  - Demonstrated that **DnR outperforms traditional neuron labeling techniques** by enhancing **concept ranking, iterative validation, and user-driven refinement**.  

## Results

Table of Scoring Methods and Neuron Metrics  

| **Neuron ID** | **Scoring Method**                     | **Reference Neuron** | **Rank** | **Score**  |
|--------------|--------------------------------|-----------------|------|---------|
| **1023**     | Top-K Squared Mean            | 927             | 1024 | 0.565151 |
| **1605**     | Mean                          | 927             | 582  | 0.037649 |
| **2233**     | Median                        | 927             | 186  | 0.011822 |
| **3981**     | Squared Mean                  | 927             | 910  | 0.011951 |
| **4125**     | Top-K Log-Weighted Activation | 927             | 30   | 0.580204 |
| **5150**     | Top-K Semantic Consistency    | 927             | 31   | 0.551183 |

- **Top-K scoring methods outperform traditional activation-based ranking techniques** in identifying functionally and semantically meaningful neurons.  
  - **Top-K Log-Weighted Activation and Top-K Semantic Consistency ranked neuron 927 in the top 30-31 neurons.**  
  - Traditional methods ranked it significantly lower (**Mean: 582, Median: 186, Squared Mean: 910**).  
  - This suggests neuron 927 exhibits **strong but sparse activations**, rather than consistently high responses across all inputs.  

- **Higher scores from Top-K methods (~0.58 and ~0.55) indicate both strong activations and semantic alignment with meaningful concepts.**  
  - Peak activations are more informative than averaging across all responses.  

- **Limitations of Traditional Methods:**  
  - **Mean and Median activations distribute importance evenly**, leading to the down-ranking of neurons with sporadic but high-impact activations.  
  - **Top-K Squared Mean ranked neuron 927 at 1024**, despite a high score of **0.565**, highlighting that neurons with even greater cumulative squared activations outperformed it.  
  - **Global averaging techniques fail to differentiate between consistently active neurons and specialized, high-impact ones.**  

- **Iterative scoring inconsistencies:**  
  - Some neurons complete in **2-3 iterations**, while others **continue for more**, leading to inconsistent convergence.  
  - **Marginal score differences** in some cases suggest redundancy in concept generation.  
  - **Stable Diffusion randomness makes quantification difficult**, as images are generated before and after step 3, with no direct comparison method to concept sets.  

- **User input concepts underperform GPT-generated concepts:**  
  - **Often scored lower than GPT-generated concepts.**  
  - **Fail to improve Stable Diffusion output**, reducing overall effectiveness in refining neuron interpretability.  

## Conclusion  

- **Top-K scoring consistently produces the most accurate results for neuron interpretability**, but it does not explain **why** certain neurons activate highly for specific images.  
  - This lack of interpretability makes it difficult to determine why **Top-K scoring identifies the best activating concepts.**  
  - Neural networks still retain aspects of the **"black box" problem**, leaving many unknowns about their inner workings.  

- **Limitations of user input concepts and iterative methods:**  
  - **User input concepts introduce redundancy** when integrated into an AI-driven generation pipeline.  
  - **Iterative refinement struggles with local minima**, leading to **non-optimal concept scoring.**  
  - The stopping threshold in iterations requires **further tuning to avoid premature termination.**  

- **Future Improvements:**  
  - **Exploring optimization algorithms** to reduce local minima stopping.  
  - **Feature tuning systems** to enhance user input integration.  
  - **Instant scoring functionality** to provide real-time feedback for user-defined concepts.  
  - Future experiments will aim to **refine these approaches** to improve neuron interpretability further.  
