import { Component, Inject, OnInit  } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import {DataServiceService} from '../data-service.service';

@Component({
  selector: 'app-facet-explanation-component',
  templateUrl: './facet-explanation-component.component.html',
  styleUrls: ['./facet-explanation-component.component.css']
})
export class FacetExplanationComponentComponent implements OnInit{

  content: any;

  constructor(@Inject(MAT_DIALOG_DATA) public data: any,private dataService: DataServiceService) { 
  }

  ngOnInit(): void {
    this.dataService.getArticleContent(this.data.source,this.data.selected_article).subscribe(data => {
      this.content = data;
    });
  }

}
