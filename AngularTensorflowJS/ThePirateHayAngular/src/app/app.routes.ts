import { Routes } from '@angular/router';
import {CamComponent} from './cam/cam.component';
import {HomeComponent} from './home/home.component';
import {TrainerComponent} from './trainer/trainer.component';
import {TfjsdetectionComponent} from './tfjsdetection/tfjsdetection.component';

export const routes: Routes = [
  { path: '', component: HomeComponent},
  { path: 'pretrained', component: CamComponent },
  { path: 'detection', component: TfjsdetectionComponent} // TODO add a url path for the specific model you want
];
