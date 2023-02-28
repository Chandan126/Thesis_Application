import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FacetExplanationComponentComponent } from './facet-explanation-component.component';

describe('FacetExplanationComponentComponent', () => {
  let component: FacetExplanationComponentComponent;
  let fixture: ComponentFixture<FacetExplanationComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ FacetExplanationComponentComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(FacetExplanationComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
