def PrepareInputs(eeg, envL, envR, attended_ear):

    # --- Align EEG and envelopes ---
    eeg_trim, env_left, env_right = align_lengths(eeg, envL, envR)

    # --- Assign attended/unattended envelopes ---
    env_att, env_unatt = get_attended(attended_ear, env_left, env_right)


    return eeg_trim, env_att, env_unatt




def align_lengths(eeg, env_left, env_right):
    """Trim or pad so EEG and envelopes have equal length."""
    #print(len(eeg), len(env_left), len(env_right))
    min_len = min(len(eeg), len(env_left), len(env_right))
    eeg=eeg[:min_len]
    env_left = env_left[:min_len]
    env_right = env_right[:min_len]
    return eeg, env_left, env_right


def get_attended(attended_ear, env_left, env_right):
    #print("attended_ear should be",str(attended_ear).upper())
    if str(attended_ear).upper().endswith("L"):

        #print("attended ear = L")
        return env_left, env_right
    else:
        #print("attended ear = R")
        return env_right, env_left



