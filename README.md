# VaM_Physical_AI_from_Mimickit
Run physical AI trained with Mimickit inside Virt-A-Mate (VaM) . Interact with the AI via VR controllers. (push and grab)

⚠️ **Disclaimer:** This is an unofficial community project. It is not affiliated with, associated with, or endorsed by Mimickit or MeshedVR.

📺 **DEMO:** [Insert your video link here]

---

## Overview

This project runs the AI inference via **Newton** in the background while running Virt-A-Mate. **A pre-trained physical AI model is included in this repository, allowing you to try it out immediately.**

* **Desktop Mode:** Works on a single PC. However, physical interactions inside Virt-A-Mate are not available in this mode.
* **VR Mode:** Running in VR mode requires heavy computing power. It will likely require a separate dedicated PC with an NVIDIA GPU just to handle the AI inference alongside the VaM PC. *(Note: A single PC setup with an RTX 4070 Ti was insufficient for VR mode during testing).*
* **Hardware Compatibility:** Developed and tested on **Meta Quest 3**. Controller tracking on other Head-Mounted Displays (HMDs) has not been verified.

---

## Network & Components

The data flows in a loop via UDP:
`VaM VRhand` ➔ `127.0.0.1 UDP 9999` ➔ `Newton` ➔ `127.0.0.1 UDP 9998` ➔ `VaM Bone`

* IP addresses and ports are fully configurable.
* If you are using a dual-PC setup, please modify the IP addresses to match your local network environment.
* **Virt-A-Mate Side:** You can change the VR hand outbound IP/port and the bone inbound port directly from the plugin UI.
* **Newton/Mimickit Side:** You can change the configurations in `data/envs/amp_g1_vam_env.yaml` under `vam_udp_ip`, `vam_udp_port`, and `vrhand_udp_port`.

### Included VaM Plugins
* `MimickitReceiver.cs` (To be loaded as a **Person plugin**)
* `VRhand_Mimickit2.cs` (To be loaded as a **Scene plugin**)

---

## Setup Instructions

1. Install **Mimickit**: [https://github.com/xbpeng/MimicKit] (Please make sure to select and configure **Newton** as the simulator backend during the installation process).
2. Follow the official Mimickit instructions to ensure it runs correctly on your system.
3. Download the latest `.zip` release from the **Releases** section of this repository and extract it.
4. Copy the contents of the extracted `MimicKit` folder into your main Mimickit installation directory. 
   * ⚠️ **Important:** This will overwrite files inside `MimicKit/mimickit/envs`. Make sure to **back up the original files** before overwriting.
5. Run the following command based on `Start_command.txt`:
   ```bash
   python mimickit/run.py --mode test --num_envs 1 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_g1_vam_env.yaml --agent_config data/agents/amp_g1_vam_agent.yaml --visualize true --model_file checkpoint/model.pt
   ```
6. In Virt-A-Mate, load `MimickitReceiver.cs` as a **Person plugin** and `VRhand_Mimickit2.cs` as a **Scene plugin**.
7. **Done!** The AI will start moving, and you can now interact with it using your hands.


---

## License & Attributions

* **Repository Code:** Distributed under the **MIT License** (applies to original repository code only).
* **Modified Mimickit Files:** The modified versions of `amp_env.py` and `task_location_env.py` contain code from the original Mimickit project and are distributed under the **Apache License 2.0**. Clear notices of the original authorship and the modifications made are documented at the top of each respective file.
