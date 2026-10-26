using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;

public class VRHandUDPSender : MVRScript
{
    // IPアドレスの4要素を保持するスライダー用変数
    private JSONStorableFloat ipOctet1JSON;
    private JSONStorableFloat ipOctet2JSON;
    private JSONStorableFloat ipOctet3JSON;
    private JSONStorableFloat ipOctet4JSON;
    private JSONStorableFloat portJSON;

    // デフォルト値 (192.168.2.119)
    public float DEFAULT_IP_1 = 127f;
    public float DEFAULT_IP_2 = 0f;
    public float DEFAULT_IP_3 = 0f;
    public float DEFAULT_IP_4 = 1f;
    public int DEFAULT_PORT = 9999;

    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;

    private const float SEND_INTERVAL = 1f / 30f;
    private float timer = 0f;
	
	private int logCounter = 0;

    public override void Init()
    {
        try
        {
            udpClient = new UdpClient();

            // 1. IPアドレス 第1オクテット (例: 192)
            ipOctet1JSON = new JSONStorableFloat("IP Octet 1", DEFAULT_IP_1, 0f, 255f, true, true);
            ipOctet1JSON.setCallbackFunction = delegate (float val) { UpdateEndPoint(); };
            UIDynamicSlider slider1 = CreateSlider(ipOctet1JSON);
            if (slider1 != null) { slider1.label = "IP Address [1]"; slider1.valueFormat = "F0"; slider1.quickButtonsEnabled = true; }
            RegisterFloat(ipOctet1JSON);

            // 2. IPアドレス 第2オクテット (例: 168)
            ipOctet2JSON = new JSONStorableFloat("IP Octet 2", DEFAULT_IP_2, 0f, 255f, true, true);
            ipOctet2JSON.setCallbackFunction = delegate (float val) { UpdateEndPoint(); };
            UIDynamicSlider slider2 = CreateSlider(ipOctet2JSON);
            if (slider2 != null) { slider2.label = "IP Address [2]"; slider2.valueFormat = "F0"; slider2.quickButtonsEnabled = true; }
            RegisterFloat(ipOctet2JSON);

            // 3. IPアドレス 第3オクテット (例: 2)
            ipOctet3JSON = new JSONStorableFloat("IP Octet 3", DEFAULT_IP_3, 0f, 255f, true, true);
            ipOctet3JSON.setCallbackFunction = delegate (float val) { UpdateEndPoint(); };
            UIDynamicSlider slider3 = CreateSlider(ipOctet3JSON);
            if (slider3 != null) { slider3.label = "IP Address [3]"; slider3.valueFormat = "F0"; slider3.quickButtonsEnabled = true; }
            RegisterFloat(ipOctet3JSON);

            // 4. IPアドレス 第4オクテット (例: 119)
            ipOctet4JSON = new JSONStorableFloat("IP Octet 4", DEFAULT_IP_4, 0f, 255f, true, true);
            ipOctet4JSON.setCallbackFunction = delegate (float val) { UpdateEndPoint(); };
            UIDynamicSlider slider4 = CreateSlider(ipOctet4JSON);
            if (slider4 != null) { slider4.label = "IP Address [4]"; slider4.valueFormat = "F0"; slider4.quickButtonsEnabled = true; }
            RegisterFloat(ipOctet4JSON);


            // 5. ポート番号の設定（既存の正常動作しているスライダー構成）
            portJSON = new JSONStorableFloat(
                "Port Number",
                DEFAULT_PORT,
                1024f,
                65535f,
                true,
                true
            );

            portJSON.setCallbackFunction = delegate (float val)
            {
                UpdateEndPoint();
            };

            UIDynamicSlider portSlider = CreateSlider(portJSON);
            if (portSlider != null)
            {
                portSlider.label = "Port Number";
                portSlider.valueFormat = "F0";
                portSlider.quickButtonsEnabled = true;
            }
            RegisterFloat(portJSON);


            // 初回のEndPoint構築
            UpdateEndPoint();
        }
        catch (Exception e)
        {
            SuperController.LogError("UDP Init Error: " + e.Message);
        }
    }

    // 4つのスライダーの値からIP文字列を組み立ててエンドポイントを更新する
    private void UpdateEndPoint()
    {
        try
        {
            if (ipOctet1JSON == null || ipOctet2JSON == null || ipOctet3JSON == null || ipOctet4JSON == null || portJSON == null) return;

            // 各スライダーのFloat値を丸めてIntに変換
            int o1 = Mathf.RoundToInt(ipOctet1JSON.val);
            int o2 = Mathf.RoundToInt(ipOctet2JSON.val);
            int o3 = Mathf.RoundToInt(ipOctet3JSON.val);
            int o4 = Mathf.RoundToInt(ipOctet4JSON.val);
            int currentPort = Mathf.RoundToInt(portJSON.val);

            // "192.168.2.119" のような文字列を動的に生成
            string currentIP = $"{o1}.{o2}.{o3}.{o4}";

            remoteEndPoint = new IPEndPoint(IPAddress.Parse(currentIP), currentPort);
            SuperController.LogMessage($"UDP Target Updated: {currentIP}:{currentPort}");
        }
        catch (Exception e)
        {
            SuperController.LogError("IP/Port Address Parse Error: " + e.Message);
        }
    }

    void Update()
    {
        timer += Time.deltaTime;
        if (timer < SEND_INTERVAL) return;
        timer = 0f;

        if (udpClient == null || remoteEndPoint == null) return;

        Transform leftHand = SuperController.singleton.leftHand;
        Transform rightHand = SuperController.singleton.rightHand;
		Transform hmdTransform = SuperController.singleton.centerCameraTarget.transform;

        if (leftHand == null || rightHand == null || hmdTransform == null) return;

        float leftTrigger = SuperController.singleton.GetLeftGrabVal();
        float rightTrigger = SuperController.singleton.GetRightGrabVal();
		
		/* logCounter++;
        if (logCounter >= 30)
        {
            Vector3 hmdPos = hmdTransform.position;
            SuperController.LogMessage($"[UDP Debug] HMD Pos: ({hmdPos.x:F2}, {hmdPos.y:F2}, {hmdPos.z:F2}) | L-Trig: {leftTrigger:F2} | R-Trig: {rightTrigger:F2}");
            logCounter = 0;
        } */

        byte[] payload = new byte[23 * 4];
        int offset = 0;

        WriteHandData(payload, ref offset, leftHand);
        CopyFloatToBuffer(payload, ref offset, leftTrigger);

        WriteHandData(payload, ref offset, rightHand);
        CopyFloatToBuffer(payload, ref offset, rightTrigger);
		
		WriteHandData(payload, ref offset, hmdTransform);

        try
        {
            udpClient.Send(payload, payload.Length, remoteEndPoint);
        }
        catch (Exception)
        {
        }
    }

    private void WriteHandData(byte[] buffer, ref int offset, Transform hand)
    {
        Vector3 pos = hand.position;
        Quaternion quat = hand.rotation;

        float posX = pos.x;
        float posY = pos.y + 0.2f; 
        float posZ = pos.z;

        float qX = quat.x;
        float qY = quat.y;
        float qZ = quat.z;
        float qW = quat.w;

        CopyFloatToBuffer(buffer, ref offset, posX);
        CopyFloatToBuffer(buffer, ref offset, posY);
        CopyFloatToBuffer(buffer, ref offset, posZ);
        CopyFloatToBuffer(buffer, ref offset, qX);
        CopyFloatToBuffer(buffer, ref offset, qY);
        CopyFloatToBuffer(buffer, ref offset, qZ);
        CopyFloatToBuffer(buffer, ref offset, qW);
    }

    private void CopyFloatToBuffer(byte[] buffer, ref int offset, float value)
    {
        byte[] bytes = BitConverter.GetBytes(value);
        Buffer.BlockCopy(bytes, 0, buffer, offset, 4);
        offset += 4;
    }

    void OnDestroy()
    {
        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
            SuperController.LogMessage("UDP Sender Closed.");
        }
    }
}
