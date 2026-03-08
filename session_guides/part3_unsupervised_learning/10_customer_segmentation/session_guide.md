# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 10: Customer Segmentation (Project)
### Single 120-Minute Session

> **The most common real-world unsupervised learning application.**
> Every e-commerce company, bank, and retailer runs this.
> K-Means + PCA + business interpretation = a complete analytics product.

---

# SESSION (120 min)
## "Segment customers — and tell each group's story"

## Before They Arrive
- Terminal open in `unsupervised_learning/projects/`
- Think of 4 different types of shoppers you know personally

---

## OPENING (10 min)

> *"How does Amazon decide who gets the 'Prime' upsell?*
> *How does Spotify decide who sees the 'upgrade' button?*
> *How does your credit card company decide who to offer a higher limit?*
>
> *They segment their customers.*
> *Cluster them into groups with similar behavior.*
> *Then design different experiences for each group.*
>
> *This is probably the most commercially valuable application*
> *of unsupervised learning.*
> *Today you build it — end to end.*
> *Data → clusters → business names → marketing strategy.*
> *This is a complete analytics deliverable."*

Write on board:
```
THE BUSINESS QUESTION:
  "We have 300 customers. Who are they?
   What do they need? How should we treat each group differently?"

OUR FEATURES:
  Age, Annual Income, Spending Score,
  Purchase Frequency, Account Age (months)

OUR TOOLS:
  Normalization → K-Means → Elbow + Silhouette → PCA (visualization)
  → Business interpretation → Marketing recommendations
```

---

## SECTION 1: Run the Pipeline (30 min)

```bash
python3 customer_segmentation.py
```

While it runs, set expectations:

> *"Watch the normalization section first.*
> *Without it: annual income (0-100,000) dominates everything.*
> *With it: all features contribute equally.*
>
> *Then the elbow and silhouette curves load.*
> *Look for where they agree on k.*
> *That's our number of segments."*

Watch the elbow curve and silhouette scores together:
> *"Where's the elbow? Where does silhouette peak?*
> *Do they agree? Usually yes. If not — silhouette wins.*
> *Let's say k=4. Four customer segments. What do they look like?"*

---

## SECTION 2: Interpreting the Clusters (25 min)

Look at the centroid table (in original unscaled units):

```
Example output:
Cluster  Age   Income    Spending  Frequency  Account Age
───────────────────────────────────────────────────────────
  0       32    95,000    78         12         48 mo
  1       55    45,000    32          5         84 mo
  2       27    38,000    61          9         12 mo
  3       44    82,000    21          3         72 mo
```

> *"Now the creative part: give each cluster a name.*
> *Based purely on the numbers, who are these people?"*

Work through with the group:
```
Cluster 0: Young (32), high income (95K), high spending (78), frequent → "High Value Loyalists"
Cluster 1: Older (55), medium income (45K), low spending (32), infrequent → "At-Risk Veterans"
Cluster 2: Young (27), lower income (38K), medium spending (61), new account → "Young Aspirationals"
Cluster 3: Middle (44), high income (82K), low spending (21), rare visits → "Untapped High-Earners"
```

> *"The untapped high-earners are gold.*
> *They have money but aren't spending it with you.*
> *What's stopping them? That's a product/marketing research question.*
> *The cluster found the opportunity. Now humans have to act."*

---

## SECTION 3: PCA Visualization (15 min)

Look at the cluster visualization in PCA 2D space:

> *"We clustered in 5D space. We can't draw that.*
> *PCA reduces to 2D for visualization — preserving 70-80% of variance.*
> *The clusters should look relatively separated here.*
>
> *One important warning: don't re-cluster in PCA space.*
> *Always cluster in the full feature space.*
> *PCA 2D is for VISUALIZATION ONLY.*
> *The clustering was done with all 5 features."*

---

## SECTION 4: Business Recommendations (20 min)

> *"A data analysis without recommendations is incomplete.*
> *Here's what you'd tell the marketing team:"*

Write on board:
```
SEGMENT → STRATEGY:

High Value Loyalists:
  → Loyalty program with premium perks
  → Early access to new products
  → Referral incentives ($50 per referral)
  → Budget: $200/customer/year

Young Aspirationals:
  → Targeted promotions (discount-driven)
  → Social media marketing (Instagram, TikTok)
  → Upgrade path: show them premium tier benefits
  → Budget: $50/customer/year

Untapped High-Earners:
  → Premium product showcasing
  → Personalized outreach (concierge service)
  → Understand BARRIERS: bad UX? Wrong selection?
  → Budget: $150/customer/year (high potential ROI)

At-Risk Veterans:
  → Win-back campaigns ("We miss you!")
  → Survey to understand churn risk
  → Tailored re-engagement (not generic blasts)
  → Budget: $75/customer/year
```

---

## CLOSING (20 min)

> *"A few questions to make this concrete:*
>
> *1. Which cluster would you want more of?*
> *(Usually: High Value Loyalists)*
>
> *2. Which cluster should worry you?*
> *(At-Risk Veterans — churning long-time customers is painful)*
>
> *3. What feature is missing from our dataset that you'd add?*
> *(Returns rate? Last purchase date? Product category?)*
>
> *These are the real business conversations that follow the analysis.*
> *The algorithm gives you the groups.*
> *Humans give them meaning and decide what to do.*"

```
CUSTOMER SEGMENTATION — COMPLETE:
  ✅ Always normalize before clustering
  ✅ Elbow + silhouette to choose k (use both)
  ✅ Centroids in original scale for interpretation
  ✅ PCA 2D for visualization (not for re-clustering)
  ✅ Business names + marketing recommendations

  NEXT: Anomaly Detection — the other side of unsupervised learning.
  Not "what groups exist?" but "what is unusual?"
```

---

## INSTRUCTOR TIPS

**"What if the clusters change every run?"**
> *"K-Means uses random initialization — results can vary.*
> *Fix: set random_state=42 for reproducibility.*
> *Also: use n_init=10 (10 random starts, keep best).*
> *sklearn's default is n_init=10 and init='k-means++' — already stable."*

**"How do I know if the segments are real?"**
> *"Three checks:*
> *1. Silhouette score > 0.4 (clean separation)*
> *2. Hierarchical clustering finds similar groups (check ARI)*
> *3. Domain experts look at the centroids and say 'yes, these match real customer types'*
> *The last one is most important — business validation beats statistical validation."*

---

## Quick Reference
```
Single Session (120 min)
├── Opening — business context         10 min
├── Run pipeline + live analysis       30 min
├── Cluster interpretation             25 min
├── PCA visualization                  15 min
├── Business recommendations           20 min
└── Closing discussion                 20 min
```

---
*MLForBeginners · Part 3: Unsupervised Learning · Module 10*
