import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Tfjsdetectionv2Component } from './tfjsdetectionv2.component';

describe('Tfjsdetectionv2Component', () => {
  let component: Tfjsdetectionv2Component;
  let fixture: ComponentFixture<Tfjsdetectionv2Component>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Tfjsdetectionv2Component]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Tfjsdetectionv2Component);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
