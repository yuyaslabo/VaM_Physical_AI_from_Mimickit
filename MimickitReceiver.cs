using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;

public class MimickitUDPReceiver : MVRScript
{
    private UdpClient _udpClient;
    private JSONStorableFloat _portParam;
	public int UDP_PORT = 9998;

    // ============================================================
    // Configuration
    // ============================================================

    private const int BONE_COUNT = 19;

    // 19 bones
    // 3 position floats + 4 quaternion floats
    // 19 * 7 * 4 = 532 bytes
    private const int PACKET_SIZE = 532;

    private const float TORSO_SMOOTH_SPEED = 70.0f;
    private const float LIMB_SMOOTH_SPEED = 70.0f;

    private readonly string[] _targetIDs = {
    "abdomen2Control",
    "chestControl",
    "headControl",
    "rHandControl",
    "lHandControl",
    "rFootControl",
    "lFootControl",
    "rElbowControl",
    "lElbowControl",
    "rKneeControl",
    "lKneeControl",
    "rArmControl",
    "lArmControl",
    "rThighControl",
    "lThighControl",
    "rShoulderControl",
    "lShoulderControl",
    "rToeControl",
    "lToeControl"
	};

    // ============================================================
    // VaM controllers
    // ============================================================

    private Transform[] _bones =
        new Transform[BONE_COUNT];

    // ============================================================
    // Receive buffers
    // ============================================================

    private Vector3[] _receivePos =
        new Vector3[BONE_COUNT];

    private Quaternion[] _receiveRot =
        new Quaternion[BONE_COUNT];

    // ============================================================
    // Apply buffers
    // ============================================================

    private Vector3[] _applyPos =
        new Vector3[BONE_COUNT];

    private Quaternion[] _applyRot =
        new Quaternion[BONE_COUNT];

    // ============================================================
    // State
    // ============================================================

    private bool _hasReceivedData = false;

    // ============================================================
    // Reusable float buffer
    //
    // Important:
    // Do NOT allocate a new byte[4] every time.
    // This buffer is reused for every float.
    // ============================================================

    private byte[] _floatBytes =
        new byte[4];

    // ============================================================
    // Initialization
    // ============================================================

    public override void Init()
    {
        _portParam = new JSONStorableFloat(
            "Port Number",
            UDP_PORT,
            1024f,
            65535f,
            true,
            true
        );

        _portParam.setCallbackFunction =
            delegate(float val)
            {
                RestartUDP();
            };

        UIDynamicSlider slider =
            CreateSlider(_portParam);

        if (slider != null)
        {
            slider.valueFormat = "F0";
            slider.quickButtonsEnabled = true;
        }

        RegisterFloat(_portParam);

        // --------------------------------------------------------
        // Find VaM controllers
        // --------------------------------------------------------

        if (containingAtom != null &&
            containingAtom.freeControllers != null)
        {
            foreach (FreeControllerV3 ctrl
                in containingAtom.freeControllers)
            {
                if (ctrl == null)
                    continue;

                int index =
                    GetBoneIndex(ctrl.name);

                if (index >= 0)
                {
                    _bones[index] =
                        ctrl.transform;

                    _receivePos[index] =
                        ctrl.transform.localPosition;

                    _receiveRot[index] =
                        ctrl.transform.localRotation;

                    _applyPos[index] =
                        ctrl.transform.localPosition;

                    _applyRot[index] =
                        ctrl.transform.localRotation;

                    // ------------------------------------------------
                    // Hand rotation
                    // ------------------------------------------------

                    if (ctrl.name == "headControl")
                    {
                        JSONStorableFloat rotForce =
                            ctrl.GetFloatJSONParam(
                                "holdRotationMaxForce"
                            );

                        if (rotForce != null)
                            rotForce.val = 50f;

                        JSONStorableFloat rotSpring =
                            ctrl.GetFloatJSONParam(
                                "holdRotationSpring"
                            );

                        if (rotSpring != null)
                            rotSpring.val = 50f;
                    }
					
					if (ctrl.name == "rHandControl" ||
                        ctrl.name == "lHandControl")
                    {
                        JSONStorableFloat rotForce =
                            ctrl.GetFloatJSONParam(
                                "holdPositionMaxForce"
                            );

                        if (rotForce != null)
                            rotForce.val = 400f;

                        JSONStorableFloat rotSpring =
                            ctrl.GetFloatJSONParam(
                                "holdPositionSpring"
                            );

                        if (rotSpring != null)
                            rotSpring.val = 400f;
                    }

                    // ------------------------------------------------
                    // Controller state
                    // ------------------------------------------------

                    ctrl.currentPositionState =
                        FreeControllerV3.PositionState.On;

                    ctrl.currentRotationState =
                        FreeControllerV3.RotationState.On;

                    // Arm controls disabled
                    if (ctrl.name == "rArmControl" || ctrl.name == "lArmControl" || ctrl.name == "rShoulderControl" || ctrl.name == "lShoulderControl" ||  ctrl.name == "rToeControl" || ctrl.name == "lToeControl")
                    {
                        ctrl.currentPositionState =
                            FreeControllerV3.PositionState.Off;

                        ctrl.currentRotationState =
                            FreeControllerV3.RotationState.Off;
                    }
					if (ctrl.name == "rHandControl" || ctrl.name == "lHandControl" || ctrl.name == "rFootControl" || ctrl.name == "lFootControl"|| ctrl.name == "rElbowControl" || ctrl.name == "lElbowControl" 
					|| ctrl.name == "rKneeControl" || ctrl.name == "lKneeControl")
                    {
                        ctrl.currentPositionState =
                            FreeControllerV3.PositionState.On;

                        ctrl.currentRotationState =
                            FreeControllerV3.RotationState.Off;
                    }
                }
                else
                {
                    ctrl.currentPositionState =
                        FreeControllerV3.PositionState.Off;

                    ctrl.currentRotationState =
                        FreeControllerV3.RotationState.Off;
                }
            }
        }

        RestartUDP();
    }

    // ============================================================
    // Bone index
    // ============================================================

    private int GetBoneIndex(string id)
	{
		if (id == "abdomen2Control") return 0;
		if (id == "chestControl") return 1;
		if (id == "headControl") return 2;

		if (id == "rHandControl") return 3;
		if (id == "lHandControl") return 4;

		if (id == "rFootControl") return 5;
		if (id == "lFootControl") return 6;

		if (id == "rElbowControl") return 7;
		if (id == "lElbowControl") return 8;

		if (id == "rKneeControl") return 9;
		if (id == "lKneeControl") return 10;

		if (id == "rArmControl") return 11;
		if (id == "lArmControl") return 12;

		if (id == "rThighControl") return 13;
		if (id == "lThighControl") return 14;

		if (id == "rShoulderControl") return 15;
		if (id == "lShoulderControl") return 16;

		if (id == "rToeControl") return 17;
		if (id == "lToeControl") return 18;

		return -1;
	}

    // ============================================================
    // UDP restart
    // ============================================================

    private void RestartUDP()
    {
        CloseUDP();

        try
        {
            int port =
                (int)_portParam.val;

            _udpClient =
                new UdpClient(port);

            _udpClient.BeginReceive(
                new AsyncCallback(ReceiveCallback),
                null
            );

            SuperController.LogMessage(
                "UDP Binary Receiver: Listening on port " +
                port +
                " / Packet=" +
                PACKET_SIZE +
                " bytes"
            );
        }
        catch (Exception e)
        {
            SuperController.LogError(
                "UDP Error: " +
                e.Message
            );
        }
    }

    // ============================================================
    // UDP callback
    // ============================================================

    private void ReceiveCallback(IAsyncResult ar)
    {
        try
        {
            if (_udpClient == null)
                return;

            IPEndPoint remoteEP =
                new IPEndPoint(
                    IPAddress.Any,
                    0
                );

            byte[] data =
                _udpClient.EndReceive(
                    ar,
                    ref remoteEP
                );

            // ----------------------------------------------------
            // Only accept exactly 420 byte packets
            // ----------------------------------------------------

            if (data != null &&
                data.Length == PACKET_SIZE)
            {
                ParsePacket(data);

                _hasReceivedData = true;
            }

            // ----------------------------------------------------
            // Continue receiving
            // ----------------------------------------------------

            if (_udpClient != null)
            {
                _udpClient.BeginReceive(
                    new AsyncCallback(ReceiveCallback),
                    null
                );
            }
        }
        catch (Exception)
        {
            // UDP shutdown / malformed packet
        }
    }

    // ============================================================
    // Binary parser
    // ============================================================

    private void ParsePacket(byte[] data)
    {
        int offset = 0;
        int i;

        for (i = 0; i < BONE_COUNT; i++)
        {
            float px =
                ReadFloat(data, offset);
            offset += 4;

            float py =
                ReadFloat(data, offset);
            offset += 4;

            float pz =
                ReadFloat(data, offset);
            offset += 4;

            float qx =
                ReadFloat(data, offset);
            offset += 4;

            float qy =
                ReadFloat(data, offset);
            offset += 4;

            float qz =
                ReadFloat(data, offset);
            offset += 4;

            float qw =
                ReadFloat(data, offset);
            offset += 4;

            _receivePos[i] =
                new Vector3(
                    px,
                    py,
                    pz
                );

            _receiveRot[i] =
                new Quaternion(
                    qx,
                    qy,
                    qz,
                    qw
                );
        }
    }

    // ============================================================
    // Compatible float parser for old .NET
    //
    // Little Endian IEEE754 float32
    //
    // BitConverter.Int32BitsToSingle()
    // is NOT used because VaM's old .NET
    // does not provide it.
    // ============================================================

    private float ReadFloat(
        byte[] data,
        int offset)
    {
        _floatBytes[0] =
            data[offset];

        _floatBytes[1] =
            data[offset + 1];

        _floatBytes[2] =
            data[offset + 2];

        _floatBytes[3] =
            data[offset + 3];

        return BitConverter.ToSingle(
            _floatBytes,
            0
        );
    }

    // ============================================================
    // Unity Update
    // ============================================================

    
	public void Update()
	{
		if (!_hasReceivedData)
			return;

		if (containingAtom == null)
			return;

		if (containingAtom.mainController == null)
			return;

		// --------------------------------------------------------
		// Copy latest received frame
		// --------------------------------------------------------

		int i;

		for (i = 0; i < BONE_COUNT; i++)
		{
			_applyPos[i] = _receivePos[i];
			_applyRot[i] = _receiveRot[i];
		}

		_hasReceivedData = false;

		// --------------------------------------------------------
		// Root transform
		// --------------------------------------------------------

		Transform rootT =
			containingAtom.mainController.transform;

		Quaternion rootRotation =
			rootT.rotation;

		// --------------------------------------------------------
		// Apply to VaM
		// --------------------------------------------------------

		for (i = 0; i < BONE_COUNT; i++)
		{
			Transform bone = _bones[i];

			if (bone == null)
				continue;

			// Torso: abdomen, chest, head
			// Limbs: hands, feet, elbows, knees, arms, thighs,
			//        shoulders, toes
			float smoothSpeed =
				(i <= 2)
					? TORSO_SMOOTH_SPEED
					: LIMB_SMOOTH_SPEED;

			float smooth = Mathf.Clamp01(
				Time.deltaTime * smoothSpeed
			);

			Vector3 worldTargetPos =
				rootT.TransformPoint(_applyPos[i]);

			bone.position = Vector3.Lerp(
				bone.position,
				worldTargetPos,
				smooth
			);

			Quaternion worldTargetRot =
				rootRotation * _applyRot[i];

			bone.rotation = Quaternion.Slerp(
				bone.rotation,
				worldTargetRot,
				smooth
			);
		}
	}


    // ============================================================
    // Cleanup
    // ============================================================

    private void CloseUDP()
    {
        if (_udpClient != null)
        {
            try
            {
                _udpClient.Close();
            }
            catch (Exception)
            {
            }

            _udpClient = null;
        }
    }

    public void OnDestroy()
    {
        CloseUDP();
    }
}