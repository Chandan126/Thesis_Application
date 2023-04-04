import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FeatureWeightingComponentComponent } from './feature-weighting-component.component';

describe('FeatureWeightingComponentComponent', () => {
  let component: FeatureWeightingComponentComponent;
  let fixture: ComponentFixture<FeatureWeightingComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ FeatureWeightingComponentComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FeatureWeightingComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
