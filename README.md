# Emner fra mastergraden

I dette repoet finner dere kode fra noen av emnene jeg har hatt, samt fra masteroppgaven min. For enkelhetens skyld har jeg ikke inkludert hele prosjektet, men kun hovedkoden.

## master_thesis

Denne mappen inneholder noe av den viktigste koden fra masteroppgaven min. Enkelt forklart, ideen bak masteroppgaven min var å finne en lett måte å inkludere nye roboter med ulike egenskaper i et system for fjernstyring av roboter. Ideen var at alle nye roboter skal etterligne den originale roboten som fjernkontrollen var implementert for, slik at de utfører samme bevegelser til tross for at de har ulike egenskaper.

### get_joint_angles.py

Denne koden henter inn bevegelsesdata fra den originale roboten og skriver til fil.

### features.py

Denne koden inneholder algoritmer for hvordan bevegelsesdataen skal bli oversatt fra en robot til en annen med andre egenskaper. Jeg utforsket tre ulike metoder, blant annet en mapping-metode der vinkelverdier blir oversatt til å passe den nye robotens egenskaper.

### misty.py

Denne koden er basert på API-et til den nye roboten og bruker de oversatte verdiene til å sende bevegelseskommandoer til roboten.


## deep_learning_for_image_analysis

Dette er et emne der jeg brukte dype nevrale nettverk for å klassifisere bilder basert på kategorien deres. Koden er skrevet i Python og bruker rammeverkene PyTorch og scikit-learn. Rapporten viser riktig klassifisering av testbilder fra noen av kategoriene.

## robot-control

Dette er et emne der jeg skrev enkel C++ kode for å styre en robot i simulering og i virkeligheten. Denne koden er basert på Robot Operating System (ROS) for robot kontrol og implementerer ulike kontrollalgoritmer.

### node.cpp

Denne koden implementerer visual servoing i en simulator, som er basert på å styre en robot ved bruk av et påmontert kamera.

### trajectory_planning.cpp

Denne koden implementerer en enkel baneplanleggingsalgoritme basert på gitte formler. Koden får inn posisjonsinformasjon fra en fysisk Turtlebot-robot og sender nye kommandoer til denne for at den skal kunne følge en viss bane.
