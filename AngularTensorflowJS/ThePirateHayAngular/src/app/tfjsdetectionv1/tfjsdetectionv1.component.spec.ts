import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Tfjsdetectionv1Component } from './tfjsdetectionv1.component';

describe('Tfjsdetectionv1Component', () => {
  let component: Tfjsdetectionv1Component;
  let fixture: ComponentFixture<Tfjsdetectionv1Component>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Tfjsdetectionv1Component]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Tfjsdetectionv1Component);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
