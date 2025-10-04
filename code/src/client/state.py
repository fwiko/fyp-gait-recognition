state = {
    "background_init": False, # Whether the gait processor's background model has been initialised
    "reset_requested": False, # Whether a reset has been requested
    "frame_counter": 0, # Number of frames since the last GEI classification attempt
    "gei_buffer": None, # Most recent GEI
    "last_saved_gei": None, # The most recent GEI transmitted
}
