import pygetwindow as gw
import pyautogui
import time

USE_SET_COORDINATES = True

def login(username, password):
    ##TODO - check if already logged in
    if USE_SET_COORDINATES:
        # Coordinates of username field
        username_field_x = 875
        username_field_y = 600

        # Click username field and type username
        pyautogui.moveTo(username_field_x, username_field_y, duration=0.5)
        pyautogui.click(username_field_x, username_field_y)
        time.sleep(0.2)
        pyautogui.typewrite(username, interval=0.05)

        # Move to password field via TAB and type password
        pyautogui.press('tab')
        time.sleep(0.2)
        pyautogui.typewrite(password, interval=0.05)

        # Submit form with Enter
        pyautogui.press('enter')
        print("Login sequence completed.")
    else:
        # Path to the image you captured of the username field
        username_image = 'username_box.png'

        # Locate the image on screen
        username_location = pyautogui.locateOnScreen(username_image)

        if username_location:
            # Get the center of the located image
            center_x, center_y = pyautogui.center(username_location)

            # Offset to the right by, say, 30 pixels
            offset_x = 60
            new_x = center_x + offset_x
            new_y = center_y

            # Click at the offset position
            pyautogui.click(new_x, new_y)
            print(f"Clicked at X={new_x}, Y={new_y}")
        else:
            print("Username field not found.")

# --- Activate the Vitrek QT Insite Window ---
def activate_qt_insite_window(window_title="Vitrek QT Insite"):
   # The part of the window title to look for
    WINDOW_TITLE = "Vitrek QT Insite"

    for window in gw.getAllWindows():
        if window.title.strip() == WINDOW_TITLE:
            qt_window = window
            break

    if not qt_window:
        print(f"Window titled '{WINDOW_TITLE}' not found.")
        exit()

    # Restore and activate it
    if qt_window.isMinimized:
        qt_window.restore()
        time.sleep(0.5)

    qt_window.activate()
    time.sleep(1)  # Give time to focus

    print(f"Activated window: {qt_window.title}")

    # Get screen size and calculate centered position
    screen_width, screen_height = pyautogui.size()
    new_left = (screen_width - qt_window.width) // 2
    new_top = (screen_height - qt_window.height) // 2
    print(f"Coordinate of top-left corner of QT Insite window: {new_left}, {new_top}")

    # Move the window to the center
    qt_window.moveTo(new_left, new_top)
    time.sleep(0.5)  # Wait for it to move

def verify_connection():
    system_config_x = 510
    system_config_y = 400

    verify_button_x = 800
    verify_button_y = 865

    print("Verifying USB connection")
    pyautogui.moveTo(system_config_x, system_config_y, MOVE_DURATION)
    pyautogui.click()

    pyautogui.moveTo(verify_button_x, verify_button_y, MOVE_DURATION)
    pyautogui.click()

    time.sleep(2)

    ##TODO
    #check if verication is successful



def run_test():
    # Coordinates of run box
    run_field_x = 850
    run_field_y = 405

    start_field_x = 1425
    start_field_y = 490

    print("Running test")
    # Click username field and type username
    pyautogui.moveTo(run_field_x, run_field_y, MOVE_DURATION)
    pyautogui.click()

    pyautogui.moveTo(start_field_x, start_field_y, MOVE_DURATION)
    pyautogui.click()
    
def track_mouse_position(interval=1):
    print("Tracking mouse position (press Ctrl+C to stop)...")
    try:
        while True:
            x, y = pyautogui.position()
            print(f"Mouse position: X={x}, Y={y}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped tracking mouse.")

# --- Main Function ---
def main():
    global MOVE_DURATION
    MOVE_DURATION = 0.5

    USERNAME = "testuser"
    PASSWORD = "AtlanticElectric"

    MODEL = "V73"
    SERIAL_NUMBER = "039402"

    ##TODO - open qt insite application if not already open

    # Activate the window
    activate_qt_insite_window()

    # # debug only - tracks mouse position
    # track_mouse_position()

    # Run login sequence
    login(USERNAME, PASSWORD)

    # Verify System Connection
    verify_connection()

    # Navigate to test screen
    run_test()

    # Export Data
    ##TODO

# --- Entry Point ---
if __name__ == "__main__":
    main()




