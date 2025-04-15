# Shared state between app.py and routes.py
state = {
    "background_init": False,
    "reset_requested": False,
    "frame_counter": 0,  # Frame counter to track frames processed
    "gei_buffer": None,  # To store the latest GEI image
    "last_saved_gei": None,  # To store the last saved GEI for comparison
}
