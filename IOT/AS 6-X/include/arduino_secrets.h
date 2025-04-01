#define SECRET_SSID "SibirienAP"
#define SECRET_PASS "Siberia51244"
#define MQTT_BROKER_ADDRESS "h5dok.cloud.shiftr.io"
#define MQTT_BROKER_PORT 8883
#define MQTT_BROKER_USERNAME "h5dok"
#define MQTT_BROKER_PASSWORD "AjZSoheCb8EAOS31"
#define MQTT_CLIENT_ID generateClientID()

String generateClientID() {
    String clientId = "theBoard-";
    clientId += String(random(0xFFFF), HEX); // Append a random hexadecimal value
    return clientId;
}