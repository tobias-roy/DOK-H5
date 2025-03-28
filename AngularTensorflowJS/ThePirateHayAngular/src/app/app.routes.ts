import { Routes } from '@angular/router';
import {CamComponent} from './cam/cam.component';
import {HomeComponent} from './home/home.component';
import {TrainerComponent} from './trainer/trainer.component';
import {TfjsdetectionComponent} from './tfjsdetection/tfjsdetection.component';
import {TfjsdetectionComponentV1} from './tfjsdetectionv1/tfjsdetectionv1.component';
import {TfjsdetectionComponentV2} from './tfjsdetectionv2/tfjsdetectionv2.component';

export const routes: Routes = [
  { path: '', component: HomeComponent},
  { path: 'pretrained', component: CamComponent },
  { path: 'detection/resnet', component: TfjsdetectionComponent},
  { path: 'detection/mobilev1', component: TfjsdetectionComponentV1},
  { path: 'detection/mobilev2', component: TfjsdetectionComponentV2}
];
