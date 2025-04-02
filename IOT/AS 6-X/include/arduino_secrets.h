#define SECRET_SSID "SibirienAP"
#define SECRET_PASS "Siberia51244"

// SHIFTR MQTT Broker
// #define MQTT_BROKER_ADDRESS "h5dok.cloud.shiftr.io"
// #define MQTT_BROKER_PORT 8883
// #define MQTT_BROKER_USERNAME "h5dok"
// #define MQTT_BROKER_PASSWORD "AjZSoheCb8EAOS31"

// HiveMQ MQTT Broker
#define MQTT_BROKER_ADDRESS "0467f9a26bb6475c88f2f8eaed4d6597.s1.eu.hivemq.cloud"
#define MQTT_BROKER_PORT 8883
#define MQTT_BROKER_USERNAME "MKRWIFI1010"
#define MQTT_BROKER_PASSWORD "TestCluster1"

#define MQTT_CLIENT_ID generateClientID()

String generateClientID() {
    String clientId = "theBoard-";
    clientId += String(random(0xFFFF), HEX); // Append a random hexadecimal value
    return clientId;
}