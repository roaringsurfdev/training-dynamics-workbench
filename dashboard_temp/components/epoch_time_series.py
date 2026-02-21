# This component encapsulates behavior for plots that show data across epochs.
# Behavior:
#   Loading:
#       If there is a selected epoch > 0, updates the plot to the selected epoch
#   Clicking on the x-axis:
#       Populates the shared Epoch left nav elements with the nearest epoch checkpoint
#       Shows a vertical line with epoch label marking the selected epoch on all plots of this type
#       Updates renders of single-epoch plots to show the newly selected epoch
#       Saves the selected epoch to state
# Might need a registry to manage all controls of this type so that they can all be
#   managed as a group