# Feedback for ws2425.2.3/team639

**Team members:** up89uvox-2

**Total points:** 58

**General notes:**
1. Please reach out if you think there was a mistake with the grading.
2. If the feedback below provides suggestions how your submission could be improved, this is not a request that you actually update your submission (unless stated otherwise). However, you might use some of the suggestions in future assignments.

-----------------

Your agent’s performance in the 3rd environment gives you **45 points**. I saw different strategies, which I found interesting, but there is definitely room for improvement. For example, in this run (https://aisysproj.kwarc.info/run/ws2425-wquest-3/6418995), it calculates the risk as more than 30% and immediately exits, but there’s 1 gold piece with a very low risk that could be collected. Maybe crossing the bridge for other gold increases the overall risk, but if you want to use this kind of strategy, it might be better to look at the risk for 1 gold, then 2 gold, and so on, rather than an average risk.  

Another strategy or possible bug is that if the plan goes back to the start position, it exits if `collected_gold > 0`. That means you miss potential rewards:  
```
if current_pos == start_pos and collected_gold > 0:
    return 'EXIT'
```
Finally, I didn’t see any improvement in the code for avoiding risk after collecting more gold. For example, if you have no gold, maybe you should be more willing to take risks. But if you already have 2 gold, you might not want to go for the 3rd. So the risk logic should also consider how much gold you have.  

There is also a ratio for skill allocation, but the strategy could be improved. For example, you need at least 3 skills for fighting to make a difference. If you only assign 2 skills, that’s effectively useless.  

In the README, the “How to run” part is good, but the “How to solution” part only briefly describes your solution. I want to see how you solved the assignment, which steps you took, what challenges you faced, and so on. That way, I can understand your thinking process—why you did certain things, how you dealt with problems, and what could be improved. This also makes it easier for me to see how your strategy can be developed further. Because this “How to solution” part is too brief and doesn’t include these details, you get **13 points** for the README.  
**Total points: 58.**
