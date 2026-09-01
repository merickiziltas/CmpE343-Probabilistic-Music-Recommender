# Tunes Duel $\boldsymbol{d}$ : A Probabilistic Simulation of Music Recommendation Systems 

CmpE 343: Introduction to Probability and Statistics, Fall 2025
Instructors: İnci M. Baytaş, Lale Akarun
TAs: Gökçe Uludoğan
Mehmet Bora Sarıoğlu
Salih Can Erer
Data Submission Due: November 5, 2025, 23:59
Code Submission Due: December 14, 2025, 23:59
Report Submission Due: December 21, 2025, 23:59

## Overview

This course project for an Introduction to Probability and Statistics class is a four-part exploration of probability and statistics concepts through the lens of a music recommendation system. You will simulate and analyze how users interact with a music recommender, using a real dataset of songs and user rating sessions. The dataset (tracks.csv, containing 2065 songs) provides metadata for each track (such as artist, release year, and an explicit content flag). Over the project, you will progressively build models to predict user preferences, design recommendation strategies, and evaluate their performance using statistical analysis.

The project is structured into four parts, each emphasizing different probabilistic tools:

- Part 1: Conditional Probability Modeling: Estimate how likely a song is to receive a $5 \star$ rating by computing smoothed conditional probabilities based on track features such as artist, year, and popularity.
- Part 2: User Variability Modeling: Model how many recommendations it takes for a user to rate a song $5 \star$, using geometric and Beta-geometric distributions.
- Part 3: Recommender Design: Develop two different recommendation models informed by the insights from Parts 1 and 2.
- Part 4: Monte Carlo Evaluation: Simulate user interactions to compare your two recommendation models, computing metrics like Hit@k, average rating, and time-to-5 $\star$, with confidence intervals.

There will be a one-week online competition following the project deadline, where anonymized submissions of your recommendation models will compete against each other. During the competition, you will be able to input a few song ratings, and each model will recommend five ranked songs in response. Models will be compared pairwise, and participants will choose which recommendation set performs better. The models' rankings will update dynamically, and surprise gifts await the top-performing teams!

## Dataset Description

You are provided with a dataset derived from Spotify's most-streamed tracks and user interactions. It combines real-world song metadata with simulated user ratings to create a mini recommendation environment.

- tracks.csv - contains metadata for 1,496 unique songs with detailed musical, artist, and audio information:
- Basic Track Info: track_id, track_name, track_popularity, duration_ms, explicit, and available_markets_count.
- Artist \& Album: artist_names, primary_artist_name, album_name, album_release_year, and ID (ab_mbid).
- Audio Features: high-level descriptors such as danceability, mood (happy, relaxed, sad, acoustic, electronic, party, aggressive), vocal gender, voice/instrumental ratio, timbre (bright/dark), and genre classification using two taxonomies (Dortmund and Rosamerica).
- ratings.csv - user-song interactions, containing:
- user_id, round_idx, song_id (matching track_id), and rating ( $1-5 \star$ ).

The dataset simulates realistic listening sessions with users of diverse musical tastes. You are expected to simulate a session by randomly selecting and listening to tracks, then recording your own ratings for the upcoming in-class competition. Each student should prepare a personal session file - in the same format as ratings.csv-with ratings for at least $\mathbf{1 0}$ and at most $\mathbf{2 0}$ songs. Please submit your session file by November 5.

## Important: Session File

Your personal session file must be determined before starting Part 1. This file will be used throughout the project for:

- Computing session-level probabilities (Part 1),
- Analyzing time-to-favorite patterns (Part 2),
- Building and testing recommendation models (Parts 3 \& 4).

Format: Match the structure of ratings.csv, with $10-20$ songs from tracks.csv and your own $1-5 \star$ ratings.

All submitted data will be anonymized and shared with the class, allowing everyone to train and validate models on collective, real user preferences.

## Part 1: Conditional Probability Modeling ( 25 points)

d "You can't always get what you want... but you can model the odds." -The Rolling Stones, gently re-phrased

In this part, you will estimate how likely a song is to receive a $5 \star$ rating, both globally (using all user data) and individually (using your own recorded session). You will analyze feature-based conditional probabilities and compare global trends with personal preferences.

## Tasks

1. Global conditional probabilities: Using tracks.csv and ratings.csv, estimate probabilities such as:

$$
P(5 \star \mid \text { Artist }=A), \quad P(5 \star \mid \text { Year }=Y), \quad P(5 \star \mid \text { Explicit }=E) .
$$

Incorporate high-level acoustic and mood features (danceability, mood, genre, timbre, etc.) as you see fit. Apply Laplace smoothing to handle sparse categories (rare artists or feature combinations).
2. Feature interactions: Explore meaningful feature combinations, such as

$$
P(5 \star \mid \text { Artist }=A, \text { Genre }=G), \quad P(5 \star \mid \text { Mood_Happy }=\text { True, Timbre }=\text { Bright }),
$$

and discuss which patterns best predict user preferences.
3. Bayesian interpretation: Use Bayes' rule to invert probabilities where informative, e.g.

$$
P(\text { Artist }=A \mid 5 \star)=\frac{P(5 \star \mid \text { Artist }=A) P(\text { Artist }=A)}{P(5 \star)},
$$

to identify which artists or audio characteristics dominate among high-rated songs.
4. Personal and group analysis: Compute analogous probabilities for your own session data:

$$
P_{m}(5 \star \mid F),
$$

where $m$ denotes your session and $F$ a chosen feature. Compare your results with your teammates' probabilities and construct a group-level model by averaging:

$$
P_{\text {group }}(5 \star \mid F)=\frac{1}{M} \sum_{m=1}^{M} P_{m}(5 \star \mid F)
$$

Discuss which attributes - artists, genres, moods, or timbres - are collectively favored.

## Discussion

Reflect on your findings:

- Which artists, release years, or popularity levels have the highest global $P(5 \star)$ ?
- Do global probabilities align with your personal favorites?
- Which musical or audio features (e.g., danceable, relaxed, electronic, vocal) seem most influential in high ratings?
- How do personal sessions differ from global and group-level tendencies?


## Part 2: User Variability Modeling ( 25 points)

d "All we need is just a little patience."
-Guns N' Roses

In Part 2, you will analyze how different users require different numbers of recommendations to find a song they love (i.e. rate $5 \star$ ). For each user (in simulated sessions), define $T_{u}$ as the number of recommendation rounds until the user gives a $5 \star$ rating. You will model the distribution of $T_{u}$ across users using two approaches:

## Modeling Tasks

- Geometric model: First, assume each recommendation has a constant probability $p$ of being a $5 \star$ "hit" for any user. Under this assumption, $T_{u}$ follows a geometric distribution: $P\left(T_{u}=t\right)=(1-p)^{t-1} p$ for $t=1,2,3, \ldots$. You will estimate the parameter $p$ from the data (e.g., via the sample mean of $T_{u}$ ) and examine how well this model fits the users' behavior.
- Beta-geometric model: Next, recognise that not all users have the same hit probability. Some users are pickier than others. To capture this variability, assume each user $u$ has their own success probability $p_{u}$, which is drawn from a $\operatorname{Beta}(\alpha, \beta)$ distribution. This hierarchical model leads to a Beta-geometric distribution for $T_{u}$ when averaged over all users. You will explore how different choices of $\alpha$ and $\beta$ can model a wider range of behaviours (e.g., many users finding a favourite quickly vs. many users requiring more trials).

You may simply discard users who never give a 5 star rating within the session limits.

## Hypothesis Testing

To formally compare user groups, perform a statistical test of differences in time-to-favorite:

- Define two or more groups, such as:
- users favoring popular vs. less popular tracks,
- users favoring older vs. newer releases,
- or members of your own team.
- Formulate hypotheses such as:

$$
H_{0}: \mu_{A}=\mu_{B} \quad \text { vs. } \quad H_{1}: \mu_{A} \neq \mu_{B},
$$

where $\mu_{A}$ and $\mu_{B}$ are mean waiting times ( $E\left[T_{u}\right]$ ) for the two groups.

- Apply an appropriate test:
- a two-sample $t$-test if data are approximately normal,
- or a nonparametric alternative such as the Mann-Whitney $U$ test if distributions are skewed.
- Report the $p$-value and interpret the result: Does one group appear to find favorites significantly faster? Discuss whether the observed difference is both statistically and practically meaningful.


## Discussion

Summarize your findings:

- How do the geometric and Beta-geometric models differ in describing user variability?
- What do the fitted parameters $(p)$ or $(\alpha, \beta)$ suggest about user patience or selectiveness?
- Were any significant differences found between user groups in your hypothesis tests?


## Part 3: Recommender Design (20 points)

d "We built this model on rock and roll."
-Starship, with minor regularization

In Part 3, you will design and implement two distinct recommendation algorithms for the music system. This is your opportunity to translate your probabilistic models into actionable recommendation strategies. You should build upon the insights from Parts 1 and 2, using the global and session-level conditional probabilities you derived, as well as the user variability patterns and patience estimates obtained from the time-to- $5 \star$ modeling.

## Objectives

Your goal is to construct and compare two different recommendation models that embody contrasting design philosophies (e.g., personalized vs. global, deterministic vs. probabilistic). Each model should take as input a few songs and ratings from the user and output a ranked list of five recommendations.

## Suggested Approaches

Possible design strategies include:

- Conditional Filtering: A personalized model that recommends songs similar to what the user has liked before. For example, it can prioritize tracks sharing the same artist, decade, or popularity group that received high ratings, effectively filtering the candidate pool using conditional probabilities from Part 1.
- Popularity-Biased: A non-personalized or lightly personalized model that emphasizes global popularity. This model may suggest songs that are highly rated in the overall dataset, ensuring universally appealing tracks are recommended first. Personalization can be added subtly e.g., by excluding disliked artists-while maintaining popularity as the dominant signal.
- Utility-Based Sampling: A probabilistic model that assigns each candidate track $i$ a utility score $U_{u}(i)$, such as

$$
U_{u}(i) \propto P_{u}(5 \star \mid i),
$$

estimated from your conditional models. Tracks are then sampled proportionally to $U_{u}(i)$, blending exploration (diversity) and exploitation (high-probability hits). You may also incorporate user patience estimates from Part 2 e.g., recommending riskier or more exploratory songs to "patient" users with higher expected $T_{u}$.

## Implementation Notes

- Clearly specify how each model computes its recommendation scores or sampling probabilities.
- Explain how insights from Part 1 (feature-based probabilities) and Part 2 (user variability) shaped your design choices.


## Evaluation in the Online Arena (Tunes Duel\&)

After submission, the best models will enter the online Tune Duel competition. The model should be implemented as a Python module compatible with the Tune Duel platform:

- Place your code under src/recommender.py.
- Implement the callable function:

```
def query(song_ratings, topk):
    ...
```

which returns a ranked list of (track_id, track_name) tuples.

- The song_ratings input provides each song's title, rating ( $1-5$ ), and track ID.
- Each model must complete inference within 8 seconds (enforced by the web platform timeout). To meet this requirement, we recommend precomputing model weights or parameters before submission and loading them during inference instead of computing them from scratch.

Users will rate a few songs, and the system will display recommendations from two anonymized models side-by-side with embedded Spotify players. Users vote for the better set, and model scores are updated using the Elo rating system. The leaderboard will update in real time throughout the one-week tournament.

## Expected Deliverables

- Two well-documented recommendation models with mathematical descriptions and implementation summaries.
- An explanation of how each model integrates the probabilistic reasoning learned earlier.
- Preliminary internal testing demonstrating that each model runs correctly on sample user inputs.


## Part 4: Monte Carlo Evaluation (20 points)

d "Is this the real life? Is this just simulation?"
-Queen, Monte Rhapsody

The final part of the project focuses on evaluating the performance of your two recommendation models from Part 3 using statistical simulation. You will conduct a Monte Carlo experiment by simulating a large number of user interactions - either by replaying existing synthetic sessions or by generating new simulated users-under each recommendation model. The objective is to obtain statistically reliable comparisons between models across several key performance metrics.

## Evaluation Metrics

You will compute and compare the following metrics for both models:

- Hit@k: For a chosen $k$ (e.g., $k=5$ or 10 ), this metric measures the proportion of users who achieve at least one $5 \star$ hit within the first $k$ recommendations. It indicates how efficiently each model surfaces songs that users love.
- Average Rating: The mean rating assigned to the recommended songs, averaged over all users and recommendation rounds (or over the first $k$ rounds). Higher average ratings suggest higher overall satisfaction and recommendation quality.
- Time-to- $\mathbf{5} \boldsymbol{*}$ : The number of recommendations (rounds) it takes for a user to give their first $5 \star$ rating. This corresponds to the random variable $T_{u}$ introduced in Part 2. Comparing the mean or median $T_{u}$ values across models helps determine which system finds "favorite" songs faster.


## Monte Carlo Simulation Procedure

1. Simulate user interactions: For each model, simulate a large number of user sessions (e.g., 1,000-5,000 trials). In each simulation, generate or replay ratings for the sequence of recommended songs.
2. Estimate performance metrics: For each trial, compute Hit@k, Average Rating, and Time-to- $5 \star$. Aggregate the results across trials to estimate the expected performance for each model.
3. Quantify uncertainty: Compute confidence intervals (e.g., 95\%) for the difference in each metric between the two models. For example:

$$
\text { Hit@5 difference: } \hat{\mu}_{A}-\hat{\mu}_{B}=0.05 \quad[0.02,0.08]
$$

indicating that Model A outperforms Model B by $5 \%$ with a $95 \%$ confidence interval of $[+2 \%,+8 \%]$.

## Interpretation and Reporting

In your evaluation report:

- Include comparative plots or tables showing model performance across metrics (e.g., bar plots, boxplots, or error bars for confidence intervals).
- Clearly state which model performs better on each metric and quantify by how much.
- Discuss whether the differences are statistically significant (based on the confidence intervals) and whether they are practically meaningful. For example: "Model $A$ achieves higher Hit@5 by focusing on personalized picks, while Model B yields slightly higher average ratings by recommending more universally popular tracks."
- Reflect on how your results relate to the design choices in Part 3. Did your probabilistic or popularity-based assumptions translate into measurable advantages?


## Deliverables

Your final report should include:

- A concise explanation of your Monte Carlo setup and assumptions.
- Visual summaries (plots or tables) of model performance.
- Confidence interval calculations for all key metrics.
- A short discussion interpreting the results in light of user behavior models from Parts 1-2.


## Submission

All projects will be submitted through both Moodle and GitHub Classroom. Each team will have its own GitHub repository created from a common project template. To join the classroom and generate your repository, please follow this invitation link. Select your student ID, and create a new team if one does not already exist, or join your existing team if it has already been created. A repository named project-[team-name] will then be generated automatically. Your GitHub repository should contain all implementation files, analysis, and supporting scripts, while the written report along with the implementation should be uploaded to Moodle. Each submission must include:

- A complete, well-documented repository containing code for all four parts.
- A short README.md file summarizing your implementation and usage instructions.
- A final project report describing your methods, results, and interpretations (a ETEX template is provided in the repository).
Code submissions are due by 14 December, 23:59. At this stage, each team must push their final recommender implementations to their assigned GitHub repository. These repositories will be used to deploy your models into the online Tune Duel competition, which will run for one week immediately following the submission deadline. The competition platform is available at this repository. Please ensure that your repository follows the required structure and implementation by testing your code as if you were another team.

Final Submission and Report: By 21 December, 23:59, teams must submit both their final code and their written project report on Moodle. The Moodle submission should include a link to your GitHub repository and the final PDF report.

## Demo Sessions

Project demonstrations will take place on 22-23 December. Each team will present and discuss its project, showcasing the recommender system, explaining its design rationale. Attendance and participation in the demo sessions are mandatory.

## Grading

The project grade will be based on both your implementation and your written analysis. There will be a short demo session, contributing to $10 \%$ of the total grade, where teams will briefly present and discuss their results. The remaining $90 \%$ will be determined by the submitted code and report.

Evaluation will be part-based, with the following approximate breakdown:

- Part 1 (Conditional Probability Modeling) - 25\%: Correct and well-documented implementation of conditional probability calculations with Laplace smoothing. Clarity in interpreting global versus individual trends and justifying predicted favorite tracks.
- Part 2 (User Variability Modeling) - 25\%: Depth and correctness of the analysis of the $T_{u}$ data, including proper estimation of geometric and Beta-geometric models and hypothesis testing between user groups.
- Part 3 (Recommender Design) - 20\%: Creativity, soundness, and justification of the two recommendation strategies. Clear explanation of how probabilistic insights from earlier parts informed the model design.
- Part 4 (Monte Carlo Evaluation) - 20\%: Rigor and completeness of the simulation experiments, including correct computation of metrics and confidence intervals. Quality of visualizations (plots, tables) and clarity of interpretation in the final report.

All team members are expected to contribute meaningfully. Grading will also consider the clarity, organization, and readability of your code and written explanations. Aim for concise but well-structured submissions that demonstrate both technical correctness and conceptual understanding and most importantly, enjoy exploring these ideas through your own creative recommender systems!

