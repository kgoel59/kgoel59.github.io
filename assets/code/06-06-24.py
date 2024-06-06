"""Hand of Straights"""

# Alice has some number of cards and she wants to rearrange the cards into groups so that each group is of size groupSize, and consists of groupSize consecutive cards.

# Given an integer array hand where hand[i] is the value written on the ith card and an integer groupSize, return true if she can rearrange the cards, or false otherwise.

 

# Example 1:

# Input: hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
# Output: true
# Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8]
# Example 2:

# Input: hand = [1,2,3,4,5], groupSize = 4
# Output: false
# Explanation: Alice's hand can not be rearranged into groups of 4.

from collections import Counter
from typing import List

class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        hand.sort()
        hand_count = Counter(hand)
        card_remaining = len(hand)

        while card_remaining >= groupSize:
            size = 0
            check = False
            prev = -1
            for keys in hand_count:
                if hand_count[keys] > 0:
                    if prev == -1 or prev == (keys - 1):
                        prev = keys
                        hand_count[keys] -= 1
                        size += 1
                        card_remaining -= 1

                        if size == groupSize:
                            check = True
                            break
                    else:
                        return False

            if not check:
                return False

        if card_remaining == 0:
            return True
        else:
            return False


if __name__ == "__main__":
    hand = [1,2,3,6,2,3,4,7,8]
    group_size = 3

    print(Solution().isNStraightHand(hand,group_size))