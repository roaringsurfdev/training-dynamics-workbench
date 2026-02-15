REQ_??? - Migrate Training and Analysis Run functionality to Dash

This requirement needs to happen before REQ_039.

Create Training and Analysis Run Job Management UI interfaces in the new dashboard (dashboard_v2). These should replace the Train tab and the top of the Analysis tab on the previous dashboard. 

If possible, it might be ideal to access these from a drop-down/site-level navigation element that might be broken down into:

Jobs/Processing/Runs
    - Training
    - Analysis
Analysis/Reports
    - Create Analysis/Report
    - Edit Analysis/Report
    - ...

Not for this requirement: Extend the existing Run Training/Run Analysis funtionality from the original dashboard to include more fully-fledged job management functionality.

That said, I think that Training and Analysis Runs will likely look very similar from a UI/UX perpective down the road.

Make sure to move the version.py over or add support for tracking different versions for the API and Consumers (dashboards, etc)