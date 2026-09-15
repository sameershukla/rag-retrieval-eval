"""Sample corpus for the chunking comparison.

Two kinds of document on purpose:

  structured   markdown headings mark where topics change
  prose        continuous text, no headings, topics drift mid paragraph

A corpus of only one kind cannot show a difference between strategies.
"""

DOCUMENTS = [
    {
        "id": "returns-policy",
        "kind": "structured",
        "text": """# Returns Policy

## Return window
Returns are accepted within 30 days of delivery. The window starts on the date the carrier marks the parcel as delivered, not the date of purchase. Orders placed during a promotional sale follow the same 30 day window.

## Condition requirements
Items must be unused and in original packaging. Items that show wear, missing accessories, or removed tags are refused at the warehouse and returned to the sender at the sender's cost. Sealed software and opened consumables cannot be returned under any circumstances.

## Refund timing
Refunds are issued to the original payment method within 5 business days of the warehouse receiving the item. Card refunds may take a further 3 to 5 days to appear depending on the issuing bank.
""",
    },
    {
        "id": "shipping-policy",
        "kind": "structured",
        "text": """# Shipping Policy

## Domestic delivery
Standard domestic delivery takes 3 to 5 business days. Express delivery arrives the next business day if the order is placed before 2pm local time.

## International delivery
International orders take 10 to 21 business days. Customs duties are paid by the recipient and are not included in the checkout total.

## Lost parcels
A parcel is treated as lost after 14 days with no carrier scan. We reship lost parcels at no cost once the carrier confirms the loss.
""",
    },
    {
        "id": "warranty",
        "kind": "structured",
        "text": """# Warranty

## Coverage period
Hardware carries a 24 month warranty from the date of delivery. Batteries and cables are covered for 6 months only.

## What is excluded
Accidental damage, liquid damage, and damage from unauthorised repair are not covered. A warranty claim requires the original order number.

## Claim process
Claims are opened through the support portal. Approved claims are repaired or replaced within 15 business days of the unit arriving at the service centre.
""",
    },
    {
        "id": "support-call-2291",
        "kind": "prose",
        "text": """Call opened at 09:14. Customer says the unit stopped charging about a week after delivery and they have been using the cable that came in the box. I checked the order and delivery was confirmed on the 3rd so we are inside the hardware window. Worth noting the cable itself is only covered for 6 months which does not matter here but I mentioned it anyway. Customer asked whether they could just return it instead of claiming warranty and I explained the return window had closed because delivery was more than 30 days ago. They were not happy about that. I offered the warranty route and they agreed. Opened claim WC-4417 in the portal and told them fifteen business days from when it lands at the service centre. Customer then asked about the parcel they sent back last month, the one with no scans, and I said we treat those as lost after fourteen days and reship at no cost, so I have queued the reship. Call closed 09:31, customer satisfied by the end.
""",
    },
    {
        "id": "support-call-2314",
        "kind": "prose",
        "text": """Customer called about an international order placed on the 12th that still has not arrived. Explained international takes ten to twenty one business days so they are still inside the normal range and there is nothing to chase yet. They then asked why the invoice did not include the customs charge and I explained duties are paid by the recipient and are not part of the checkout total, which they had not realised. Slightly tense for a minute. They also wanted to know whether they could return the item once it arrives given it will be well past the order date, and I confirmed the thirty day window runs from delivery and not from purchase, so they will have the full window once it lands. That resolved it. Mentioned in passing that the item needs to be unused and in the original packaging if they do decide to send it back. No ticket opened.
""",
    },
    {
        "id": "incident-notes-0918",
        "kind": "prose",
        "text": """Rough notes from the ingestion incident. The nightly job reported success but the document count in the index did not move. Turned out the parser was returning empty text for the newer PDF exports and we were indexing empty strings without a check, so every one of those documents was technically present and completely unsearchable. That took most of the morning to find because the job logs were green the whole time. Added a guard that fails the batch when extracted text is under fifty characters. Separately we noticed the reindex takes a lot longer since we moved to embedding every sentence for the boundary detection, roughly four times the previous wall clock, which nobody had budgeted for. Worth raising before we roll it out to the full corpus. Also the chunk sizes are now variable which is making the context budget harder to predict downstream.
""",
    },
]

# Questions paired with the exact evidence a correct answer needs.
#
# Ground truth is at the evidence level, not the chunk level. Chunk ids change
# with every strategy, so they cannot be compared across strategies. The
# sentences that must reach the model do not change.
QUESTIONS = [
    {
        "id": "q1",
        "query": "Can I return an item I have already used?",
        "evidence": [
            "Returns are accepted within 30 days of delivery.",
            "Items must be unused and in original packaging.",
        ],
    },
    {
        "id": "q2",
        "query": "When does the return window start, purchase or delivery?",
        "evidence": [
            "Returns are accepted within 30 days of delivery.",
            "The window starts on the date the carrier marks the parcel as delivered, not the date of purchase.",
        ],
    },
    {
        "id": "q3",
        "query": "How long is the warranty on a charging cable?",
        "evidence": [
            "Hardware carries a 24 month warranty from the date of delivery.",
            "Batteries and cables are covered for 6 months only.",
        ],
    },
    {
        "id": "q4",
        "query": "My warranty claim was approved, how long until I get the unit back?",
        "evidence": [
            "Claims are opened through the support portal.",
            "Approved claims are repaired or replaced within 15 business days of the unit arriving at the service centre.",
        ],
    },
    {
        "id": "q5",
        "query": "How long before a missing parcel counts as lost?",
        "evidence": [
            "A parcel is treated as lost after 14 days with no carrier scan.",
            "We reship lost parcels at no cost once the carrier confirms the loss.",
        ],
    },
    {
        "id": "q6",
        "query": "Do I pay customs on an international order?",
        "evidence": [
            "International orders take 10 to 21 business days.",
            "Customs duties are paid by the recipient and are not included in the checkout total.",
        ],
    },
    {
        "id": "q7",
        "query": "What happened in the ingestion incident and what did it cost us?",
        "evidence": [
            "the parser was returning empty text for the newer PDF exports",
            "roughly four times the previous wall clock",
        ],
    },
    {
        "id": "q8",
        "query": "On call 2291 why was the customer refused a return?",
        "evidence": [
            "the return window had closed because delivery was more than 30 days ago",
            "Opened claim WC-4417 in the portal",
        ],
    },
    {
        "id": "q9",
        "query": "How long does express domestic delivery take?",
        "evidence": [
            "Standard domestic delivery takes 3 to 5 business days.",
            "Express delivery arrives the next business day if the order is placed before 2pm local time.",
        ],
    },
    {
        "id": "q10",
        "query": "How quickly is a refund paid back to my card?",
        "evidence": [
            "Refunds are issued to the original payment method within 5 business days of the warehouse receiving the item.",
            "Card refunds may take a further 3 to 5 days to appear depending on the issuing bank.",
        ],
    },
]
