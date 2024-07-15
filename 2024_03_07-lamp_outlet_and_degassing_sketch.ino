const int PIN_RELAY_DEGASS = 8; //definition of GPIO pin for switching relay connected to degassing valve
const int PIN_RELAY_LAMP_OUTLET = 9; //definition of GPIO pin for switching relay connected to USB connection of master/slave outlet
bool isOn = false;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(PIN_RELAY_DEGASS, OUTPUT);
  pinMode(PIN_RELAY_LAMP_OUTLET, OUTPUT);
 digitalWrite(PIN_RELAY_LAMP_OUTLET, HIGH); //lamp outlet turned off
 digitalWrite(PIN_RELAY_DEGASS, HIGH); //degassing valve turned off
}

void loop() {
  // put your main code here, to run repeatedly:
  // if a paket of information is available at the serial port read it
 if (Serial.available() > 0) {

    String msg = Serial.readString();

    if (msg == "1") {
      digitalWrite(PIN_RELAY_DEGASS, LOW); //turn degassing valve on
  } 
  
    if (msg =="0") {
      digitalWrite(PIN_RELAY_DEGASS, HIGH); //turn degassing valve off
    }

     if (msg == "ON") {
       digitalWrite(PIN_RELAY_LAMP_OUTLET, LOW); //turn USB port on --> also turns outlet on
     } 
     if (msg=="OFF") {
      digitalWrite(PIN_RELAY_LAMP_OUTLET, HIGH); //turn USB port off --> also turns outlet off
     }

  }
}
