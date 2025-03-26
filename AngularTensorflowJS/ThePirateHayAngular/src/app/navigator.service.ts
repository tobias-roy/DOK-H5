import { Injectable } from '@angular/core';
import {Router} from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class NavigatorService {

  constructor(private router: Router) { }

  public NavigateTo(s: string) {
    this.router.navigate([s])
  }
}
