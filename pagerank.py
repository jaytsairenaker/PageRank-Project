#!/usr/bin/python3

'''
This file calculates pagerank vectors for small-scale webgraphs.
See the README.md for example usage.
'''

import math
import torch
import gzip
import csv
import numpy

import logging

class WebGraph():

    def __init__(self, filename, max_nnz=None, filter_ratio=None):
        '''
        Initializes the WebGraph from a file.
        The file should be a gzipped csv file.
        Each line contains two entries: the source and target corresponding to a single web link.
        This code assumes that the file is sorted on the source column.
        '''

        self.url_dict = {}
        indices = []

        from collections import defaultdict
        target_counts = defaultdict(lambda: 0)

        # loop through filename to extract the indices
        logging.debug('computing indices')
        with gzip.open(filename,newline='',mode='rt') as f:
            for i,row in enumerate(csv.DictReader(f)):
                if max_nnz is not None and i>max_nnz:
                    break
                import re
                regex = re.compile(r'.*((/$)|(/.*/)).*')
                if regex.match(row['source']) or regex.match(row['target']):
                    continue
                source = self._url_to_index(row['source'])
                target = self._url_to_index(row['target'])
                target_counts[target] += 1
                indices.append([source,target])

        # remove urls with too many in-links
        if filter_ratio is not None:
            new_indices = []
            for source,target in indices:
                if target_counts[target] < filter_ratio*len(self.url_dict):
                    new_indices.append([source,target])
            indices = new_indices

        # compute the values that correspond to the indices variable
        logging.debug('computing values')
        values = []
        last_source = indices[0][0]
        last_i = 0
        for i,(source,target) in enumerate(indices+[(None,None)]):
            if source==last_source:
                pass
            else:
                total_links = i-last_i
                values.extend([1/total_links]*total_links)
                last_source = source
                last_i = i

        # generate the sparse matrix
        # a tensor is a multidimensional array (like a NumPy array)
        i = torch.LongTensor(indices).t()
            # contains long integers
        v = torch.FloatTensor(values)
            # can do math operations w/ values
        n = len(self.url_dict)
        self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
            # putting together the values and the indices of the matrix

        self.index_dict = {v: k for k, v in self.url_dict.items()}
    

    def _url_to_index(self, url):
        '''
        given a url, returns the row/col index into the self.P matrix
        '''
        if url not in self.url_dict:
            self.url_dict[url] = len(self.url_dict)
        return self.url_dict[url]


    def _index_to_url(self, index):
        '''
        given a row/col index into the self.P matrix, returns the corresponding url
        '''
        return self.index_dict[index]


    def make_personalization_vector(self, query=None):
        '''
        If query is None, returns the vector of 1s.
        If query contains a string,
        then each url satisfying the query has the vector entry set to 1;
        all other entries are set to 0.
        '''
        n = self.P.shape[0]
        if query is None:
            v = torch.ones(n)
        else:
            v = torch.zeros(n)
            # FIXME: implement Task 2
            # for each index in the personalization vector:

            for index in range(n):
                # get the url for the index 
                url = self._index_to_url(index)
                # check if the url satisfies the input query
                if url_satisfies_query(url, query):
                    # set the corresponding index to one
                    v[index]=1            

        v_sum = torch.sum(v)
        assert(v_sum>0)
        v /= v_sum

        return v

        '''
        Task 2, part 1:
        $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
        
        INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
        INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
        INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
        INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
        INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
        INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
        INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
        INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
        INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
        INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response

        Task 2, part 2:
        $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'

        INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
        INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
        INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
        INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
        INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
        INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
        INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
        INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
        INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
        INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
        '''


    def power_method(self, v=None, x0=None, alpha=0.85, max_iterations=1000, epsilon=1e-6):
        '''
        This function implements the power method for computing the pagerank.
        The self.P variable stores the $P$ matrix.
        You will have to compute the $a$ vector and implement Equation 5.1 from "Deeper Inside Pagerank."
        '''
        with torch.no_grad():
            n = self.P.shape[0]
            # grabs the dimensions of the nxn matrix P
            # create variables if none given
            if v is None:
                v = torch.Tensor([1/n]*n)
                v = torch.unsqueeze(v,1)
                # adds a new dimension of size 1 to a tensor at a specified position
            v /= torch.norm(v)
            # applies the L2 norm to the vector v

            if x0 is None:
                x0 = torch.Tensor([1/(math.sqrt(n))]*n)
                x0 = torch.unsqueeze(x0,1)
                # adding another dimension, so this can go in sparse.addmm
                # 1xnx1??
            x0 /= torch.norm(x0)
            # normalizing x0
        
            # compute the new x vector using Eq (5.1)
                # FIXME: Task 1
                # HINT: this can be done with a single call to the `torch.sparse.addmm` function,
                # but you'll have to read the code above to figure out what variables should get passed to that function
                # and what pre/post processing needs to be done to them
            # main loop
            xprev = x0
            x = xprev.detach().clone()
            # creating the new vector AKA x^k
            
            # creating the vector a, whose elements are a 1 if dangling node, else 0 
            stochastic_rows = torch.sparse.sum(self.P, 1).indices()
            a = torch.ones([n,1])
            a[stochastic_rows] = 0
            
            for i in range(max_iterations):
                xprev = x.detach().clone()
                # calculating the second half of equation 5.1
                equation_tail = (alpha*x.t() @ a + (1-alpha))*v.t()
                # matrix multiplication and addition
                x = self.P.t().matmul(x).mul_(alpha).add_(equation_tail.t())
                # normalizing the vector
                x /= torch.norm(x)

                # output debug information
                residual = torch.norm(x-xprev)
                logging.debug(f'i={i} residual={residual}')
                # early stop when sufficient accuracy reached
                if residual < epsilon:
                    break
            #x = x0.squeeze()
            return x.squeeze()
        
        '''
        Task 1, part 1:
        DEBUG:root:i=0 residual=0.2562914788722992
        DEBUG:root:i=1 residual=0.11841226369142532
        DEBUG:root:i=2 residual=0.07070129364728928
        DEBUG:root:i=3 residual=0.03181539848446846
        DEBUG:root:i=4 residual=0.020496614277362823
        DEBUG:root:i=5 residual=0.01010835450142622
        DEBUG:root:i=6 residual=0.006371526513248682
        DEBUG:root:i=7 residual=0.0034228116273880005
        DEBUG:root:i=8 residual=0.002087965374812484
        DEBUG:root:i=9 residual=0.0011749409604817629
        DEBUG:root:i=10 residual=0.0007013162248767912
        DEBUG:root:i=11 residual=0.00040323587018065155
        DEBUG:root:i=12 residual=0.00023796527239028364
        DEBUG:root:i=13 residual=0.00013811791723128408
        DEBUG:root:i=14 residual=8.111781789921224e-05
        DEBUG:root:i=15 residual=4.723723031929694e-05
        DEBUG:root:i=16 residual=2.7683758162311278e-05
        DEBUG:root:i=17 residual=1.6175998098333366e-05
        DEBUG:root:i=18 residual=9.440776921110228e-06
        DEBUG:root:i=19 residual=5.51695256945095e-06
        DEBUG:root:i=20 residual=3.198200147380703e-06
        DEBUG:root:i=21 residual=1.911825393108302e-06
        DEBUG:root:i=22 residual=1.128756821344723e-06
        DEBUG:root:i=23 residual=6.653998525507632e-07
        INFO:root:rank=0 pagerank=6.6270e-01 url=4
        INFO:root:rank=1 pagerank=5.2179e-01 url=6
        INFO:root:rank=2 pagerank=4.1434e-01 url=5
        INFO:root:rank=3 pagerank=2.3175e-01 url=2
        INFO:root:rank=4 pagerank=1.8590e-01 url=3
        INFO:root:rank=5 pagerank=1.6917e-01 url=1
        
        Task 1, part 2:
        $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
        INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
        INFO:root:rank=1 pagerank=8.9231e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
        INFO:root:rank=2 pagerank=7.0396e-04 url=www.lawfareblog.com/britains-coronavirus-response
        INFO:root:rank=3 pagerank=6.9159e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
        INFO:root:rank=4 pagerank=6.7047e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
        INFO:root:rank=5 pagerank=6.6262e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
        INFO:root:rank=6 pagerank=6.5051e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
        INFO:root:rank=7 pagerank=6.3625e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
        INFO:root:rank=8 pagerank=6.1254e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
        INFO:root:rank=9 pagerank=6.0193e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
        
        $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
        INFO:root:rank=0 pagerank=5.7828e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
        INFO:root:rank=1 pagerank=5.2341e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
        INFO:root:rank=2 pagerank=5.1299e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
        INFO:root:rank=3 pagerank=4.6601e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
        INFO:root:rank=4 pagerank=4.5936e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
        INFO:root:rank=5 pagerank=4.3073e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
        INFO:root:rank=6 pagerank=4.0937e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
        INFO:root:rank=7 pagerank=3.7593e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
        INFO:root:rank=8 pagerank=3.4510e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
        INFO:root:rank=9 pagerank=3.4486e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

        $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
        INFO:root:rank=0 pagerank=4.5748e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
        INFO:root:rank=1 pagerank=4.4176e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
        INFO:root:rank=2 pagerank=2.6929e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
        INFO:root:rank=3 pagerank=1.9393e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
        INFO:root:rank=4 pagerank=1.5453e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
        INFO:root:rank=5 pagerank=1.5358e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
        INFO:root:rank=6 pagerank=1.5259e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
        INFO:root:rank=7 pagerank=1.4222e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
        INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
        INFO:root:rank=9 pagerank=1.1464e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force

        Task 1, part 3:
        $ python3 pagerank.py --data=data/lawfareblog.csv.gz
        INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
        INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
        INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
        INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
        INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
        INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
        INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
        INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
        INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
        INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/topics

        $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
        INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
        INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
        INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
        INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
        INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
        INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
        INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
        INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
        INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
        INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
        
        Task 1, part 4:
        $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
        INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
        INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
        INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
        INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
        INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
        INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
        INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
        INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
        INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
        INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/topics

        $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
        INFO:root:rank=0 pagerank=2.8859e-01 url=www.lawfareblog.com/snowden-revelations
        INFO:root:rank=1 pagerank=2.8859e-01 url=www.lawfareblog.com/lawfare-job-board
        INFO:root:rank=2 pagerank=2.8859e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
        INFO:root:rank=3 pagerank=2.8859e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
        INFO:root:rank=4 pagerank=2.8859e-01 url=www.lawfareblog.com/subscribe-lawfare
        INFO:root:rank=5 pagerank=2.8859e-01 url=www.lawfareblog.com/topics
        INFO:root:rank=6 pagerank=2.8859e-01 url=www.lawfareblog.com/masthead
        INFO:root:rank=7 pagerank=2.8859e-01 url=www.lawfareblog.com/our-comments-policy
        INFO:root:rank=8 pagerank=2.8859e-01 url=www.lawfareblog.com/upcoming-events
        INFO:root:rank=9 pagerank=2.8859e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
        
        $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
        INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
        INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
        INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
        INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
        INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
        INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
        INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
        INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
        INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
        INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
        
        $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
        INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
        INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
        INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
        INFO:root:rank=3 pagerank=3.1757e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
        INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
        INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
        INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
        INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
        INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
        INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
        '''


    def search(self, pi, query='', max_results=10):
        '''
        Logs all urls that match the query.
        Results are displayed in sorted order according to the pagerank vector pi.
        '''
        n = self.P.shape[0]
        vals,indices = torch.topk(pi,n)

        matches = 0
        for i in range(n):
            if matches >= max_results:
                break
            index = indices[i].item()
            url = self._index_to_url(index)
            pagerank = vals[i].item()
            if url_satisfies_query(url,query):
                logging.info(f'rank={matches} pagerank={pagerank:0.4e} url={url}')
                matches += 1


def url_satisfies_query(url, query):
    '''
    This functions supports a moderately sophisticated syntax for searching urls for a query string.
    The function returns True if any word in the query string is present in the url.
    But, if a word is preceded by the negation sign `-`,
    then the function returns False if that word is present in the url,
    even if it would otherwise return True.

    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'coronavirus covid')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'coronavirus')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid -speech')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', 'covid -corona')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '-speech')
    False
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '-corona')
    True
    >>> url_satisfies_query('www.lawfareblog.com/covid-19-speech', '')
    True
    '''
    satisfies = False
    terms = query.split()

    num_terms=0
    for term in terms:
        if term[0] != '-':
            num_terms+=1
            if term in url:
                satisfies = True
    if num_terms==0:
        satisfies=True

    for term in terms:
        if term[0] == '-':
            if term[1:] in url:
                return False
    return satisfies


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--personalization_vector_query')
    parser.add_argument('--search_query', default='')
    parser.add_argument('--filter_ratio', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=0.85)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--max_results', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    g = WebGraph(args.data, filter_ratio=args.filter_ratio)
    v = g.make_personalization_vector(args.personalization_vector_query)
    pi = g.power_method(v, alpha=args.alpha, max_iterations=args.max_iterations, epsilon=args.epsilon)
    g.search(pi, query=args.search_query, max_results=args.max_results)
 