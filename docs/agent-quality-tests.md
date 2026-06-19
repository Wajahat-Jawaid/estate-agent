# Agent Quality Tests

These are quality benchmarks for the real estate AI agent.

These are not scripts to memorize. The agent should generalize from the customer intent category and preserve the intended behavior.

## Core Sales Journey

qualification → guidance → shortlist → comparison → objection handling → contact capture → visit booking → agent handoff → follow-up

## Important Rule

Not every scenario needs to complete the full journey. Each scenario should pass the correct stage of the journey.

---

# Stage-Specific Scenarios

## 1. Family buyer

Customer:
Looking for a 3 bed apartment for my family.

Expected behavior:

* Ask budget or preferred area.
* Treat it as family living, not investment.
* Mention that area, building condition, lift, parking, and surroundings matter.
* Do not show properties immediately unless budget + area are already clear.

Pass criteria:

* Asks only one main question.
* Does not dump listings too early.
* Sounds consultative, not robotic.

## 2. School-driven family

Customer:
I need a flat mainly because of my kids’ school.

Expected behavior:

* Ask which school area or route matters.
* Prioritize commute and daily routine.
* Later shortlist based on school access, lift, parking, and family environment.
* Do not randomly suggest far areas.

Pass criteria:

* Identifies school commute as primary priority.
* Does not jump directly to listings.

## 3. Elderly parent / accessibility

Customer:
My mother lives with us and she cannot take stairs.

Expected behavior:

* Treat lift and backup power as must-haves.
* Ask about floor preference or lift backup.
* Avoid recommending buildings without confirmed lift access.
* Mention accessibility naturally.

Pass criteria:

* Prioritizes lift/backup over cosmetic features.

## 4. Investor

Customer:
I want to buy something mainly for rental income.

Expected behavior:

* Ask budget and preferred area/type.
* Focus on rental demand, tenant profile, resale, and maintenance.
* Do not guide like a family-use buyer.
* Avoid guaranteed ROI claims.

Pass criteria:

* Uses investment language.
* Does not promise exact returns unless data exists.

## 5. Premium buyer

Customer:
I want something premium, ideally DHA or Clifton.

Expected behavior:

* Ask budget and bedrooms.
* Keep tone premium and concise.
* Mention that exact phase/block and building quality matter.
* Do not push low-end areas unless budget mismatch requires expectation setting.

Pass criteria:

* Does not suggest irrelevant budget areas.

## 6. Low-budget buyer

Customer:
I need a good 3 bed flat but my budget is limited.

Expected behavior:

* Ask exact budget if not already provided.
* Set expectations politely without making the customer feel poor.
* Explain the main trade-offs: area, size, building age, location quality, amenities, and distance from central areas.
* If budget is tight for the desired area/property type, do not jump directly to secondary filters like floor, lift, drawing room, or west-open.
* First ask which compromise is acceptable:

  * keep 3-bed strict but accept older/weaker building or less prime block
  * consider a better 2-bed in a stronger building/location
  * expand to slightly farther areas
  * increase budget if possible
* Do not say nothing is possible unless inventory/data clearly supports that.
* Do not overpromise premium-area options within a tight budget.

Pass criteria:

* Helpful and respectful.
* Does not overpromise.
* Identifies the budget-vs-location-vs-size mismatch.
* Asks a trade-off question before asking secondary preference questions.


## 7. Unrealistic demand

Customer:
I want a 3 bed apartment in DHA under 1 crore.

Expected behavior:

* Politely explain that this may not be realistic.
* Offer alternatives: increase budget, compromise on area, reduce bedrooms, or consider different locations.
* Do not pretend matching properties exist.

Pass criteria:

* Corrects expectation without sounding rude.

## 8. Just browsing

Customer:
Show me some flats.

Expected behavior:

* Ask one qualifying question first: area, budget, bedrooms, or purpose.
* Do not show random properties immediately.
* Keep response short.

Pass criteria:

* Avoids random listing dump.

## 9. Area confusion

Customer:
I’m confused between Gulshan and Johar.

Expected behavior:

* Compare briefly based on family living, access, budget, congestion, and building options.
* Ask what matters more: commute, peaceful environment, newer building, or budget.
* Do not give a huge essay.

Pass criteria:

* Gives useful local guidance.
* Ends with one clear next question.

## 10. Price objection

Customer:
This seems expensive.

Expected behavior:

* Acknowledge concern.
* Explain whether price is due to area, building condition, floor, amenities, or availability.
* Offer to find cheaper alternatives with clear trade-offs.
* Do not become defensive.

Pass criteria:

* Handles objection calmly.

## 11. Visit-ready buyer

Customer:
My wife and I can visit Friday after school time.

Expected behavior:

* Treat as hot lead.
* Confirm preferred time window.
* Ask for contact only after giving value or shortlisting.
* Prepare visit/handoff flow.

Pass criteria:

* Moves toward booking, not more unnecessary qualification.

## 12. Seller / non-buyer

Customer:
I want to sell my flat.

Expected behavior:

* Detect seller intent.
* Ask location, size, bedrooms, condition, expected price.
* Do not show buyer-side property recommendations.
* Move toward valuation/listing/handoff.

Pass criteria:

* Correctly switches flow.

## 13. Post-booking message refinement

Customer:
Yes, make it slightly more polite and WhatsApp-friendly. Keep it short though.

Context:
- Visit already selected
- Property selected
- Agent known
- Customer asked for a message draft

Expected behavior:
- Refine the message using existing context.
- Do not restart property qualification.
- Do not ask what kind of property the customer is looking for.

Pass criteria:
- Produces a short WhatsApp-friendly message.
- Preserves selected property, time, agent, and concerns.
